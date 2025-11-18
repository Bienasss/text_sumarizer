from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List
import torch
from preprocessing import TextPreprocessor

class AbstractiveSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn", max_length: int = 512, min_length: int = 50):
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length
        self.preprocessor = TextPreprocessor()
        
        device = 0 if torch.cuda.is_available() else -1
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device
            )
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Falling back to t5-small")
            self.model_name = "t5-small"
            self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device
            )
    
    def _chunk_text(self, text: str, max_chunk_length: int = 1000) -> List[str]:
        sentences = self.preprocessor.segment_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > max_chunk_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def summarize(self, text: str, max_length: int = None, min_length: int = None) -> str:
        processed_text = self.preprocessor.preprocess(text)
        
        max_len = max_length or self.max_length
        min_len = min_length or self.min_length
        
        if len(processed_text.split()) <= max_len:
            try:
                result = self.summarizer(
                    processed_text,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False
                )
                return result[0]['summary_text']
            except Exception as e:
                return processed_text[:500]
        else:
            chunks = self._chunk_text(processed_text, max_chunk_length=1000)
            summaries = []
            
            for chunk in chunks:
                try:
                    result = self.summarizer(
                        chunk,
                        max_length=max_len,
                        min_length=min_len,
                        do_sample=False
                    )
                    summaries.append(result[0]['summary_text'])
                except Exception as e:
                    summaries.append(chunk[:200])
            
            return ' '.join(summaries)

class T5Summarizer(AbstractiveSummarizer):
    def __init__(self, model_name: str = "t5-small", max_length: int = 512, min_length: int = 50):
        super().__init__(model_name=model_name, max_length=max_length, min_length=min_length)
    
    def summarize(self, text: str, max_length: int = None, min_length: int = None) -> str:
        processed_text = self.preprocessor.preprocess(text)
        
        max_len = max_length or self.max_length
        min_len = min_length or self.min_length
        
        input_text = f"summarize: {processed_text}"
        
        try:
            inputs = self.tokenizer.encode(
                input_text,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_len,
                    min_length=min_len,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            return processed_text[:500]

