import base64
import mimetypes





def encode_image_b64(chart_path):
    # Guess the MIME type (e.g., image/png, image/jpeg)
    media_type, _ = mimetypes.guess_type(chart_path)
    if media_type is None:
        media_type = "application/octet-stream"  # fallback

    # Read and encode the image as base64
    with open(chart_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return media_type, encoded