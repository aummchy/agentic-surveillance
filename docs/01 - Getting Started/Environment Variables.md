# Environment Variables

All set in `.env` file at project root. Loaded by `python-dotenv` in `config/settings.py:9`.

## Secrets (env-only, never in config.jsonc)

| Variable | Default | Purpose |
|----------|---------|---------|
| `MONGODB_URI` | (required) | MongoDB Atlas connection string |
| `MONGODB_DATABASE` | `surveillance` | Database name |
| `MONGODB_COLLECTION` | `faces` | Faces collection name |
| `MONGODB_EVENTS_COLLECTION` | `events` | Events collection name |
| `CLOUDINARY_CLOUD_NAME` | `""` | Cloudinary cloud name |
| `CLOUDINARY_API_KEY` | `""` | Cloudinary API key |
| `CLOUDINARY_API_SECRET` | `""` | Cloudinary API secret |
| `ALERT_WEBHOOK_URL` | `""` | Webhook URL for alerts |
| `SMTP_HOST` | `""` | SMTP server host |
| `SMTP_PORT` | `587` | SMTP server port |
| `SMTP_USER` | `""` | SMTP username |
| `SMTP_PASS` | `""` | SMTP password |
| `ALERT_EMAIL_TO` | `""` | Alert email recipient |
| `TWILIO_ACCOUNT_SID` | `""` | Twilio account SID |
| `TWILIO_AUTH_TOKEN` | `""` | Twilio auth token |
| `TWILIO_FROM` | `""` | Twilio sender number |
| `ALERT_SMS_TO` | `""` | SMS alert recipient |

## Pipeline / Model

| Variable | Default | Purpose |
|----------|---------|---------|
| `YOLO_MODEL` | `models/yolov8s_openvino_model/` | YOLO model path (.pt or OpenVINO IR) |
| `YOLO_DEVICE` | `cpu` | Device for YOLO (`cpu`, `0` for CUDA, `intel:GPU`) |
| `INSIGHTFACE_MODEL` | `buffalo_l` | InsightFace model pack |
| `INSIGHTFACE_DET_SIZE` | `640` | Face detection input size |
| `INSIGHTFACE_PROVIDER` | `CPUExecutionProvider` | ONNX Runtime provider |
| `OPENVINO_DEVICE` | `GPU` | OpenVINO accelerator (`CPU`, `GPU`, `NPU`, `AUTO`) |

## Camera

| Variable | Default | Purpose |
|----------|---------|---------|
| `CAMERA_SOURCE` | `""` | RTSP URL or file path (overrides CAMERA_INDEX) |
| `CAMERA_INDEX` | `0` | Webcam device index |
| `CAMERA_ID` | `cam_01` | Camera identifier for composite track IDs |
| `CAMERA_BACKEND` | `""` | Video backend (`dshow`, `msmf`, or auto) |
| `FRAME_WIDTH` | `1280` | Capture width |
| `FRAME_HEIGHT` | `720` | Capture height |
| `FRAME_SKIP` | `2` | Skip every Nth frame before JPEG encode |

## LLM

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `gemma3:4b` | Model name (swap to `qwen3.5:4b`) |
| `OLLAMA_TIMEOUT` | `30` | Request timeout in seconds |

## Alerts

| Variable | Default | Purpose |
|----------|---------|---------|
| `ALERT_CHANNELS` | `console,webhook` | Comma-separated: `console`, `email`, `sms`, `webhook` |

## Override via env var

Any config.jsonc value can be overridden by env var. Example:
```bash
$env:MATCH_THRESHOLD="0.40"
$env:FRAME_SKIP="4"
```
