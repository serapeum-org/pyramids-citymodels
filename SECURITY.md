# Security Policy

## Supported versions

`pyramids-citymodels` is pre-1.0. Only the latest released version receives fixes; there are no
maintenance branches for older `0.x` releases.

| Version           | Supported          |
|-------------------|--------------------|
| latest `0.x`      | :white_check_mark: |
| any earlier `0.x` | :x:                |

## Reporting a vulnerability

Report a suspected vulnerability privately via GitHub's
[security advisory form](https://github.com/serapeum-org/pyramids-citymodels/security/advisories/new),
or by email to moah.farag@gmail.com. Please do not open a public issue for it.

Include the version, the platform, and the smallest reproduction you have. You can expect an
acknowledgement within a week. If the report is accepted, the fix ships in the next release and the
advisory is published with it; if it is declined, you get the reasoning.

Note that this package has no network or credential surface of its own — it processes local
city-model data through GDAL, which the `pyramids-core` wheel vendors. A vulnerability in GDAL itself
belongs upstream, and one in the vendoring belongs in
[pyramids](https://github.com/serapeum-org/pyramids/security).
