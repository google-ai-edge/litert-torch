<!--ts-->
* [Development Environment Setup](./CONTRIBUTING.md#development-environment-setup)
   * [Running Tests](./CONTRIBUTING.md#running-tests)
   * [Code Formatting](./CONTRIBUTING.md#code-formatting)
* [Contributor License Agreement](./CONTRIBUTING.md#contributor-license-agreement)
* [Community Guidelines](./CONTRIBUTING.md#community-guidelines)
* [Security Guidelines](./CONTRIBUTING.md#security-guidelines)
* [Code Contribution Guidelines](./CONTRIBUTING.md#code-contribution-guidelines)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->
<!-- Added by: advaitjain, at: Thu May  9 12:33:43 AM PDT 2024 -->

<!--te-->

# Development Environment Setup

Every contributor to this repository should develop in a fork.

```bash
cd litert-torch
python -m venv venv
source venv/bin/activate

pip install -r dev-requirements.txt
pip install -e .
```

## Running Tests

```bash
cd litert-torch
bash ./run_tests.sh
```

## Code Formatting

You can format your changes with our preconfigured formatting script.

```bash
cd litert-torch
bash ./format.sh
```

# Contributor License Agreement

- Contributions to this project must be accompanied by a [Contributor License
  Agreement](https://cla.developers.google.com/about) (CLA).

- Visit <https://cla.developers.google.com/> to see your current agreements or
  to sign a new one.

# Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google/conduct/).

# Security Guidelines

To maintain the security of our CI/CD pipelines, we use [Zizmor](https://github.com/woodruffw/zizmor) for static analysis of GitHub Actions workflows.

When contributing changes to GitHub workflows (`.github/workflows/*.yml`):
- Ensure all workflows pass Zizmor security scans.
- Address any findings related to unpinned actions, script injections, or excessive permissions.
- Workflow modifications may require review and approval from the infrastructure or security team.

# Code Contribution Guidelines

We recommend that contributors read these tips from the Google Testing Blog:

- [Code Health: Providing Context with Commit Messages and Bug Reports](https://testing.googleblog.com/2017/09/code-health-providing-context-with.html)
- [Code Health: Understanding Code In Review](https://testing.googleblog.com/2018/05/code-health-understanding-code-in-review.html)
- [Code Health: Too Many Comments on Your Code Reviews?](https://testing.googleblog.com/2017/06/code-health-too-many-comments-on-your.html)
- [Code Health: To Comment or Not to Comment?](https://testing.googleblog.com/2017/07/code-health-to-comment-or-not-to-comment.html)

