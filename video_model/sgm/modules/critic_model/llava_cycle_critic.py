'''
USES LLAVA_V1.5 NOT LLAVA-NEXT
'''

import torch
import numpy as np
from transformers import (
    LlavaForConditionalGeneration,  # LLaVA v1.5
    AutoProcessor,
)
from cyclereward import cyclereward
from PIL import Image
# import av


class LlavaCycleRewardCritic:
    '''
    - Uses LLaVa v1.5 model + cyclereward
    - LlaVa doesnt have video model, so process frame by frame
    - Make sure cycelreward is git clones and the ...med.py file is changed as mentione in github repo
    (github CSE219 project repo branch critic-cyclerewards)
    - Uses CycleReward to directly score frame-instruction alignment
    - Returns the CycleReward score as the reward
    '''

    def __init__(
            self,
            model_id: str = "llava-hf/llava-1.5-7b-hf",  # LLaVA v1.5
            device: str | torch.device = None,
            num_frames: int = 8,
            cr_model_type: str = "CycleReward-Combo",
        ):
            # pick device
            if device is None:
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                elif torch.backends.mps.is_available():
                    device = torch.device("mps")
                else:
                    device = torch.device("cpu")
            self.device = device

            # load LLaVA v1.5 model + processor
            print("Loading LLaVA v1.5 model...")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self.model.eval()

            self.processor = AutoProcessor.from_pretrained(model_id)
            print("LLaVA v1.5 loaded.")

            # Load CycleReward model
            print("Loading CycleReward model...")
            self.cr_model, self.cr_preproc = cyclereward(
                device=str(device), 
                model_type=cr_model_type
            )
            self.cr_model.eval()
            print("CycleReward loaded.")

            self.num_frames = num_frames
            self.reward_stats = {'min': float('inf'), 'max': float('-inf'), 'mean': 0.0, 'count': 0}
    
    def _sample_video_frames(self, video_path: str, num_frames: int):
        """Uniformly sample frames from video using PyAV"""
        try:
            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            if total_frames is None or total_frames <= 0:
                # Fallback: take first N frames
                frames = []
                for i, frame in enumerate(container.decode(video=0)):
                    if i >= num_frames:
                        break
                    frames.append(frame.to_rgb().to_ndarray())
                return np.stack(frames, axis=0)
            
            indices = np.linspace(0, max(total_frames - 1, 0), num_frames).astype(int)
            frames = []
            for i, frame in enumerate(container.decode(video=0)):
                if i in indices:
                    frames.append(frame.to_rgb().to_ndarray())
                if len(frames) == num_frames:
                    break
            return np.stack(frames, axis=0)  # (T, H, W, 3)
        except Exception as e:
            print(f"Error sampling video frames: {e}")
            # Return black frames as fallback
            return np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)

    def _run_llava_v1_5(self, frames: np.ndarray, question: str) -> str:
        """
        Run LLaVA v1.5 on frames - processes one frame at a time
        frames: (T, H, W, 3) rgb uint8
        returns generated answer string
        """
        answers = []
        
        # LLaVA v1.5 processes one image at a time
        for i in range(len(frames)):
            try:
                pil_image = Image.fromarray(frames[i])
                
                # LLaVA v1.5 prompt format
                prompt = f"USER: <image>\n{question}\nASSISTANT:"
                
                inputs = self.processor(
                    text=prompt, 
                    images=pil_image, 
                    return_tensors="pt", 
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs, 
                        max_new_tokens=100, 
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                
                answer = self.processor.decode(output[0], skip_special_tokens=True)
                # Extract just the assistant's response
                if "ASSISTANT:" in answer:
                    answer = answer.split("ASSISTANT:")[-1].strip()
                answers.append(answer)
                
            except Exception as e:
                print(f"Error running LLaVA on frame {i}: {e}")
                answers.append("")
        
        # Return the most common non-empty answer
        non_empty_answers = [a for a in answers if a.strip()]
        if non_empty_answers:
            # Simple majority voting
            from collections import Counter
            most_common = Counter(non_empty_answers).most_common(1)[0][0]
            return most_common
        else:
            return "No description generated."

    def _compute_cyclereward(self, frames: np.ndarray, instruction: str) -> float:
        """
        Compute CycleReward for frames against instruction.
        Returns mean score across frames.
        """
        scores = []
        
        for i in range(len(frames)):
            try:
                # Convert frame to PIL Image
                if frames.dtype != np.uint8:
                    frame_uint8 = (frames[i] * 255).clip(0, 255).astype(np.uint8)
                else:
                    frame_uint8 = frames[i]
                
                pil_image = Image.fromarray(frame_uint8)
                
                # Preprocess for CycleReward
                img_tensor = self.cr_preproc(pil_image).unsqueeze(0).to(self.device)
                
                # Get score
                with torch.no_grad():
                    score = self.cr_model.score(img_tensor, instruction)
                
                # Convert to float
                if isinstance(score, torch.Tensor):
                    frame_score = float(score.mean().cpu().numpy())
                elif isinstance(score, list):
                    frame_score = float(np.mean(score))
                else:
                    frame_score = float(score)
                
                scores.append(frame_score)
                
            except Exception as e:
                print(f"Error computing CycleReward for frame {i}: {e}")
                scores.append(0.0)
        
        if not scores:
            return 0.0
            
        mean_score = float(np.mean(scores))
        
        # Update statistics for normalization
        self.reward_stats['min'] = min(self.reward_stats['min'], mean_score)
        self.reward_stats['max'] = max(self.reward_stats['max'], mean_score)
        self.reward_stats['mean'] = (self.reward_stats['mean'] * self.reward_stats['count'] + mean_score) / (self.reward_stats['count'] + 1)
        self.reward_stats['count'] += 1
        
        return mean_score

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward to [0, 1] range based on observed statistics"""
        if self.reward_stats['count'] < 10:
            return reward
            
        reward_range = self.reward_stats['max'] - self.reward_stats['min']
        if reward_range > 0:
            normalized_reward = (reward - self.reward_stats['min']) / reward_range
        else:
            normalized_reward = 0.5
            
        return normalized_reward

    # --------- public APIs ---------
    def score_video(
        self,
        video_path: str,
        question: str,
        reference: str,
        normalize_reward: bool = True,
    ) -> dict:
        """
        Main entrypoint if you have a video file on disk.
        Returns dict with candidate answer and scalar reward.
        """
        frames = self._sample_video_frames(video_path, self.num_frames)
        
        # Use direct CycleReward scoring (more efficient)
        reward = self._compute_cyclereward(frames, reference)
        
        # Optional: Also run LLaVA for analysis
        answer = self._run_llava_v1_5(frames, question)
        
        if normalize_reward:
            reward = self._normalize_reward(reward)
        
        return {
            "answer": answer,
            "reward": reward,
        }

    def score_frames(
        self,
        frames: np.ndarray,
        question: str,
        reference: str,
        normalize_reward: bool = True,
        use_llava: bool = False,  # Option to skip LLaVA for efficiency
    ) -> dict:
        """
        Entry if you already have diffusion-generated video frames
        as a numpy array (T, H, W, 3), uint8 or float in [0,1].
        """
        if frames.dtype != np.uint8:
            # assume float [0,1]
            frames = (frames * 255).clip(0, 255).astype(np.uint8)
        
        if use_llava:
            # Run LLaVA first (like original approach)
            answer = self._run_llava_v1_5(frames, question)
            reward = self._compute_cyclereward(frames, answer)
        else:
            # Direct approach: CycleReward on frames vs instruction
            answer = "Direct CycleReward evaluation"
            reward = self._compute_cyclereward(frames, reference)
        
        if normalize_reward:
            reward = self._normalize_reward(reward)
        
        return {
            "answer": answer,
            "reward": reward,
        }

    def score_frames_direct(
        self,
        frames: np.ndarray,
        instruction: str,
        normalize_reward: bool = True,
    ) -> dict:
        """
        Direct scoring without LLaVA - just CycleReward on frames vs instruction.
        Most efficient for RL training.
        """
        if frames.dtype != np.uint8:
            frames = (frames * 255).clip(0, 255).astype(np.uint8)
        
        reward = self._compute_cyclereward(frames, instruction)
        
        if normalize_reward:
            reward = self._normalize_reward(reward)
        
        return {
            "answer": f"Direct CycleReward: {instruction}",
            "reward": reward,
        }

    def cleanup(self):
        """Clean up memory"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'cr_model'):
            del self.cr_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
