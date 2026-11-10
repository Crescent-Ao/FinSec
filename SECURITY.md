# Security

Do not include API keys, private code, datasets or model weights in issues or pull requests.
Teacher API credentials are read from local environment variables or protected local files.

The audit Agent treats source code, comments and tool returns as data. Python and
shell tools execute inside a networkless Bubblewrap workspace with CPU, memory,
time and file limits. Keep this isolation enabled when testing model-generated code.

For a suspected vulnerability, use GitHub private vulnerability reporting when
available, or contact a repository maintainer without posting sensitive details.
