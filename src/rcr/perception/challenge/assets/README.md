# Runtime resources

[Ours] Three trained Microban landmark checkpoints and their schemas are included.
Their hashes and the original runtime source hashes are in provenance.json.

The RCR STL files are downloaded with pinned source URLs and SHA-256 checks.
The source generator is GPL-3.0: https://github.com/jackybaltes/rcr-perception .
Microban CAD is installed from the pinned microban-model package, without copying
the third_party/microban submodule. M3T is built from its pinned public source.
Ultralytics checkpoints use the Ultralytics software; see its AGPL-3.0 licence.
No gated third-party model checkpoint is required by the selected pipelines.
