# Changelog

All notable changes to PyDSGEforge are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Clean, reproducible Dynare parity fixture and validation command.
- Medium-scale hybrid NK and small-open-economy YAML showcases.
- Community health files, issue templates, local quality commands, and package
  metadata.

### Changed

- Corrected forward-looking state augmentation, aligning the likelihood,
  posterior, and unit-shock IRFs with Dynare to numerical precision.
- Reworked Docker into separate builder, test, and non-root runtime stages.

## [0.1.0] - 2026-07-14

- Initial public research release of symbolic model construction, Gensys
  solution, Kalman likelihood, MAP, MCMC, IRFs, and YAML-first configuration.

[Unreleased]: https://github.com/pablo-reyes8/PyDSGEforge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pablo-reyes8/PyDSGEforge/releases/tag/v0.1.0
