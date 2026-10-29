
"""PyInstaller executable entry; app supplies token/model/data via environment."""
import uvicorn
from photoindex.api_v1 import create_app

if __name__ == "__main__":
    uvicorn.run(create_app(), host="127.0.0.1", port=8765,
                workers=1, reload=False, access_log=False,
                proxy_headers=False, server_header=False)
