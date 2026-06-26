# Quant Portfolio-Kaizen Desktop

The desktop shell loads the same Next.js application used by the PWA. It does
not contain analytics or portfolio logic.

Security defaults:

- `contextIsolation: true`
- `nodeIntegration: false`
- Electron sandbox enabled
- permission requests denied
- internal navigation restricted to `QPK_WEB_URL`
- external HTTPS and mail links opened by the operating system

Development:

```powershell
npm run desktop:dev
```

Production should set `QPK_WEB_URL` to the validated Vercel deployment.
