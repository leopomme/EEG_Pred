# Official public evidence snapshot

Retrieved on 3 October 2026. `*_pages.json` are complete responses from Kaggle's public page API; the Markdown files contain the corresponding published page content. `*_metadata.json` contain current public competition configuration. `organizer_messages.json` includes the host reply and current public replies; no private data or hidden solution was requested.

Ordinary web rendering was empty or failed for some Kaggle URLs. We first checked the pages with the web tool, then retrieved the same public material from documented-in-client read endpoints. Requests used no login, API token or other credentials.

Read endpoints used:

- `GET https://www.kaggle.com/api/i/competitions.CompetitionService/GetCompetition?competitionName=<slug>`
- `GET https://www.kaggle.com/api/i/competitions.PageService/ListPages?competitionId=<id>`
- `GET https://www.kaggle.com/api/i/competitions.CompetitionService/GetCompetitionDatabundleVersion?competitionId=<id>&competitionDatabundleType=COMPETITION_DATABUNDLE_TYPE_PUBLIC`
- `GET https://www.kaggle.com/api/i/discussions.DiscussionsService/GetForumTopicById?forumTopicId=743366`
- `GET https://www.kaggle.com/api/i/discussions.DiscussionsService/GetTopicListByForumId?forumId=<forum-id>`
- Read-only `POST https://www.kaggle.com/api/i/discussions.DiscussionsService/GetForumMessagesInTopic` with body `{"topicId":743366,"includeFirstForumMessage":true}`. This retrieves messages; it does not send one.

Single Subject: competition 143178, forum 11288053, public data version 21011638.
Cross Subject: competition 157329, forum 12208451, public data version 20764975.

Canonical user-facing sources:

- [Single overview/evaluation](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab/overview/evaluation)
- [Single data](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab/data)
- [Single rules](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab/rules)
- [Cross overview/evaluation](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab-cross-subject/overview/evaluation)
- [Cross data](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab-cross-subject/data)
- [Cross rules](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab-cross-subject/rules)
- [Organizer discussion](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab/discussion/743366)
- [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks#technical-specifications), read through web search/open; platform limits summarized in `notebook_platform_limits.json`.

`snapshot_manifest.json` records SHA-256 digests for the saved evidence. A live refresh can differ; keep this snapshot immutable and create a new dated snapshot when rules change.
