# Security policy

This project is an experimental Colab-hosted development bridge, not a hardened production multi-user AI service.

- Keep `NGROK_AUTHTOKEN`, `BRIDGE_PASSWORD`, and `HF_TOKEN` in Colab Secrets. Never put them in source files, screenshots, issues, or commits.
- Require a strong unique Bridge password. The public ngrok URL alone is not access control.
- Do not expose the local llama.cpp server to the public Internet. The Bridge communicates with it through `127.0.0.1`.
- If credentials leak, rotate them promptly and terminate the previous ngrok session.
- Browser-local chat history may remain accessible to people who use the same browser profile. Export or clear it intentionally.
- Running arbitrary, untrusted model files has risks. This project uses `trust_remote_code=False` for Transformers and does not intentionally execute repository scripts.
- Real estate photos and prompts may contain personal information. Review them before sharing or including in public examples.
- Model licenses differ: confirm commercial use rights for your chosen model weights before publishing generated content.

For vulnerabilities in a public repository, configure GitHub private vulnerability reporting rather than posting secrets in a public issue.
