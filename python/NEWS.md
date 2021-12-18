# `tabben` v0.0.6

## New Features
- Some datasets now have "extras" stored as JSON, which will be downloaded and used to provide extra data like data profiles, bibtex, dataset licensing info, etc.
- `OpenTabularDataset` has new properties: `has_extras`, `license`, `bibtex`

## Breaking Changes
- removed `sarcos` dataset as a result of [this article](https://www.datarobot.com/blog/running-code-and-failing-models/)

## Non-Breaking Changes
- Names of some internal modules have changed

## Bugfixes
- fixed bugs when using a non-lowercased name of a dataset for the OpenTabularDataset constructor

## Non-Code Updates
- (Temporary) [documentation website](https://umd-otb.github.io/OpenTabularDataBenchmark/)! Contains package docs as well as dataset docs.
