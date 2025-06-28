import os
import openai
from dotenv import load_dotenv
openai.api_key = os.getenv("OPENAI_API_KEY")

load_dotenv() 
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def classify_sentiment(text: str) -> dict:
    prompt = (
    f"""Analyze the following financial news headline and classify its sentiment as 
    'positive', 'neutral', or 'negative', each with a probability score.
    Headline: "{text}"
    Respond with a JSON object with keys:
      sentiment: one of positive, neutral, negative,
      probabilities: negative, positive, neutral (values between 0 and 1),
      confidence: max probability of the three values (a value between 0 and 1)"""
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial news sentiment analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0
    )
    import json
    content = response.choices[0].message.content
    try:
        result = json.loads(content)
        sentiment = result.get("sentiment", "unknown").lower()
        # Dummy confidence and probabilities since GPT does not provide them
        LABELS = ["negative", "neutral", "positive"]
        probs = [0.0, 0.0, 0.0]
        if sentiment in LABELS:
            idx = LABELS.index(sentiment)
            probs[idx] = 1.0
        else:
            sentiment = "unknown"
        return {
            "sentiment": sentiment,
            "confidence": float(max(probs)),
            "probabilities": dict(zip(LABELS, map(float, probs))),
        }
    except Exception:
         # fallback: unknown sentiment
        LABELS = ["negative", "neutral", "positive"]
        return {
            "sentiment": "unknown",
            "confidence": 0.0,
            "probabilities": dict(zip(LABELS, [0.0, 0.0, 0.0])),
        }

def batch_classify(news_list: list[str]) -> list[dict]:
    return [classify_sentiment(text) for text in news_list]

headline = "Due to world war 1, all tech stocks are down by 100%"
result = classify_sentiment(headline)
print(f"Sentiment: {result['sentiment']}")
print(result["probabilities"])
print(f"Confidence: {result['confidence']*100:.1f}%")