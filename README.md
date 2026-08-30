# Attention Trading Agent
 
An autonomous AI trading agent built for the Alpaca AI Trading Agents Hackathon.
 
## Idea
 
We combine technical and fundamental analysis through a real attention mechanism — not just prompting. The attention layer dynamically weighs multiple signals (price/volatility data, plus fresh news and earnings context via RAG) to decide what matters most moment to moment. That weighted context is passed to an LLM to make the trade or allocation call.
 
Every decision then passes through a hard-coded risk layer (position size caps, correlation exposure limits, liquidity checks, and daily loss limits) before anything executes on Alpaca
