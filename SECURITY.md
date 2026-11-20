# Security policy

## Reporting a vulnerability

Please report security problems privately. Use GitHub private vulnerability
reporting for this repository: open the
[security advisories page](https://github.com/GTLIDAR/ImitationLab/security/advisories)
and choose "Report a vulnerability". Do not open a public issue for a
suspected vulnerability.

Include the affected command or module, the steps to reproduce, and the commit
hash you tested.

## Supported versions

Security fixes are applied to the `main` branch, which is the supported
version.

## Loading checkpoints

ImitationLab loads checkpoints and encoder files with
`torch.load(..., weights_only=False)`, which unpickles arbitrary Python
objects and can execute code contained in the file. Load only checkpoints,
export bundles, and encoder files from sources you trust.
