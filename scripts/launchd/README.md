# graphrag worker backstop (launchd)

Writes (`graphrag note`, MCP adds) and recalls start `graphrag work` on their
own. This agent runs it every 15 minutes as a backstop, for writes from
surfaces that don't spawn (Nushell, Python) and for retries after an LLM
outage. A second worker exits immediately ("already running"), so overlap is
harmless.

Install:

```sh
mkdir -p ~/Library/Logs/graphrag
cp scripts/launchd/com.ndn.graphrag-work.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ndn.graphrag-work.plist
```

Remove:

```sh
launchctl bootout gui/$(id -u)/com.ndn.graphrag-work
rm ~/Library/LaunchAgents/com.ndn.graphrag-work.plist
```

Pause during big builds without unloading it: `graphrag work --pause`, then
`graphrag work --resume`. While paused every run exits at once.

Exit codes: 0 done (or already running), 64 refused (re-extraction needs
`--accept-reextract`), 69 LLM endpoint or model unavailable, 70 local error,
75 LLM outage.
