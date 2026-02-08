import os
import json
from groq import Groq
from dotenv import load_dotenv
from getdata import GetData

class SentimentAnalyze:
    def __init__(self, model = "llama-3.3-70b-versatile"):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key = api_key)
        self.model = model

    def headline_analyzer(self, headline):
        prompt = f"""
        You are a professional text analyzer. Analyze the financial seniment of {headline},
        Return valid JSON with:
        - "score": A float between -1.0 and 1.0. 
                   Use precise decimals (e.g., 0.15, -0.42, 0.95).
                   0.0 is neutral. 
                   -0.1 to -0.3 is slightly negative.
                   -0.8 to -1.0 is catastrophic (bankruptcy/crash).
        - "reason": A brief explanation (max 10 words).
        """

        try:
            response = self.client.chat.completions.create(
                messages = [{
                    "role" : "user",
                    "content" : prompt
                }],
                model = self.model,
                response_format = {"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as es:
            print(f"error: {es}")
            return None
"""      
def main():
    gd = GetData("TSLA")
    sa = SentimentAnalyze()
    news = gd.get_news_headline()
    for headline in news[:10]:
        head_txt = headline[2]
        score = sa.headline_analyzer(head_txt)
        print(score)

if __name__ == "__main__":
    main()
""" 


