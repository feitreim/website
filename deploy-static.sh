#!/usr/bin/env bash
# Sync gitignored static assets (images, fonts) to the VPS and rebuild.
#
# static/ is gitignored (licensed fonts aren't redistributed), so its files
# don't travel via `git push`. This pushes them straight to the server, then
# restarts the service so the build copies them into dist/.
set -euo pipefail

HOST="${1:-website}" # ssh host alias; see ~/.ssh/config
REMOTE_DIR="/home/blog/website/static/"

rsync -rtvz --exclude='.DS_Store' static/ "$HOST:$REMOTE_DIR"
ssh "$HOST" 'chown -R blog:blog /home/blog/website/static && systemctl restart website'
echo "synced static/ -> $HOST and restarted website"
