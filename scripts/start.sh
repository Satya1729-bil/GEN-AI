#!/bin/sh
# Container start: build the index if it is missing, then start the app.
set -e

if [ ! -f "${RX_INDEX_DIR:-index}/faiss.index" ]; then
  echo "No index found. Building it from ${RX_DATA_DIR:-data} ..."
  python -m rxreveal.build_index || echo "Index not built (no PDFs yet). Upload files in the app."
fi

exec streamlit run app.py \
  --server.port="${PORT:-8501}" \
  --server.address=0.0.0.0
