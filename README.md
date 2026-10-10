# LexBrief AI

AI-assisted legal-document summarisation and summary evaluation. The app accepts pasted text and supported document uploads, supports adjustable summary lengths, and evaluates summaries against extracted source material.

## Deploy on Render

This repository includes a Dockerfile and `render.yaml`. Create a Render Web Service from this repository using the Docker runtime. Configure `OPENAI_API_KEY` as a secret environment variable in Render. `OPENAI_MODEL` and `OPENAI_EVALUATOR_MODEL` can be adjusted as needed.

## Important

The app requires a valid OpenAI API key for AI summarisation and evaluation. Keep the key secret; do not place it in source code or commit it to GitHub. Legal summaries and evaluation results are aids for review, not a substitute for checking the original document.
