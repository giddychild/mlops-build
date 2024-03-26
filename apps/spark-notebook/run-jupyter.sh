#!/usr/bin/env bash
export PATH="$PATH:/home/jovyan/.local/bin"

cd "/home/jovyan"
exec /home/jovyan/.local/bin/jupyter "${JUPYTERTYPE:=lab}" \
  --notebook-dir="${HOME}" \
  --ip=0.0.0.0 \
  --no-browser \
  --allow-root \
  --port=8888 \
  --ServerApp.token="" \
  --ServerApp.password="" \
  --ServerApp.allow_origin="*" \
  --ServerApp.base_url="${NB_PREFIX}" \
  --ServerApp.authenticate_prometheus=False