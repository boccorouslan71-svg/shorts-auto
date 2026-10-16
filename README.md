# shorts-auto

Transforme une vidéo longue 16:9 en **3 shorts 9:16** (1080x1920) sous-titrés, avec une pub à la fin, envoyés sur Google Drive, puis prévient Make par webhook.

- Workflow : `.github/workflows/shorts.yml` (déclenché par `workflow_dispatch`)
- Script : `shorts.py` (Python + FFmpeg)
- Pub : à déposer dans `assets/pub.mp4` (sinon ignorée, avertissement dans le callback)

## Entrées
`video_url` (obligatoire), `job_id` (obligatoire), `mode` (`blur` ou `crop`, défaut `blur`), `title` (optionnel).

## Secrets (Settings > Secrets and variables > Actions)
`GROQ_API_KEY`, `RCLONE_CONFIG_B64`, `DRIVE_FOLDER_ID`, `MAKE_CALLBACK_URL`.

## Déclenchement
`POST /repos/{OWNER}/shorts-auto/actions/workflows/shorts.yml/dispatches` avec `{"ref":"main","inputs":{"video_url":"...","job_id":"...","mode":"blur"}}`.

## Callback
JSON `{job_id, status, source_url, title, parts[{index, drive_file_id, filename, duration_sec}], warnings, error}` ; `status` = `success` ou `error`.
