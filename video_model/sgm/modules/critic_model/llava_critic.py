import torch
import numpy as np
# from transformers import (
#     LlavaNextVideoProcessor,
#     LlavaNextVideoForConditionalGeneration,
# )
from transformers import (
    LlavaForConditionalGeneration,  # LLaVA v1.5
    AutoProcessor, LlavaProcessor
)
from bert_score import score


class LlavaBertCritic:
    """
    Critic that:
    1) runs LLaVA-NeXT-Video on a video + question
    2) compares the generated answer to a reference text using BERTScore
    3) returns the BERTScore F1 as the reward

    You can plug this into your diffusion/TTA loop.
    """

    def __init__(
        self,
        # model_id: str = "llava-hf/LLaVA-NeXT-Video-7B-hf",
        model_id: str = "llava-hf/llava-1.5-7b-hf",  # LLaVA v1.5
        device: str | torch.device = None,
        num_frames: int = 8,
        bert_lang: str = "en",
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

        # load LLaVA video model + processor
        # self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        #     model_id,
        #     torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
        #     low_cpu_mem_usage=True,
        # ).to(device)
        self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True,
        )
        self.model.eval()

        # self.processor = LlavaNextVideoProcessor.from_pretrained(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # BERTScore config
        self.bert_lang = bert_lang
        self.num_frames = num_frames

    # # --------- low-level helpers ---------
    # def _sample_video_frames(self, video_path: str, num_frames: int):
    #     """uniformly sample num_frames from a video path using PyAV"""
    #     container = av.open(video_path)
    #     total_frames = container.streams.video[0].frames
    #     indices = np.linspace(0, max(total_frames - 1, 0), num_frames).astype(int)

    #     frames = []
    #     container.seek(0)
    #     for i, frame in enumerate(container.decode(video=0)):
    #         if i in indices:
    #             frames.append(frame.to_ndarray(format="rgb24"))
    #         if len(frames) == num_frames:
    #             break
    #     return np.stack(frames, axis=0)  # (T, H, W, 3)

    def _build_prompt(self, question: str):
        # LLaVA chat template: one user turn with text + video
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "video"},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        return prompt

    def _run_llava(self, frames: np.ndarray, question: str) -> str:
        """
        frames: (T, H, W, 3) rgb uint8
        returns generated answer string
        """
        prompt = self._build_prompt(question)
        inputs = self.processor(
            text=prompt,
            videos=frames,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=100, do_sample=False
            )
        # skip special tokens
        answer = self.processor.decode(output[0][2:], skip_special_tokens=True)
        return answer

    def _bert_score(self, candidate: str, reference: str) -> float:
        P, R, F1 = score([candidate], [reference], lang=self.bert_lang, verbose=False)
        return F1.item()

    # --------- public APIs ---------
    def score_video(
        self,
        video_path: str,
        question: str,
        reference: str,
    ) -> dict:
        """
        Main entrypoint if you have a video file on disk.
        Returns dict with candidate answer and scalar reward.
        """
        frames = self._sample_video_frames(video_path, self.num_frames)
        answer = self._run_llava(frames, question)
        reward = self._bert_score(answer, reference)
        return {
            "answer": answer,
            "reward": reward,
        }

    def score_frames(
        self,
        frames: np.ndarray,
        question: str,
        reference: str,
    ) -> dict:
        """
        Entry if you already have diffusion-generated video frames
        as a numpy array (T, H, W, 3), uint8 or float in [0,1].
        """
        if frames.dtype != np.uint8:
            # assume float [0,1]
            frames = (frames * 255).clip(0, 255).astype(np.uint8)
        answer = self._run_llava(frames, question)
        reward = self._bert_score(answer, reference)
        return {
            "answer": answer,
            "reward": reward,
        }
