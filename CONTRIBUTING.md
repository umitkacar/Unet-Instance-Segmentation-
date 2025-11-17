# Contributing to Awesome U-Net Instance Segmentation

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Submission Guidelines](#submission-guidelines)
- [Code Style Guidelines](#code-style-guidelines)
- [Adding New Resources](#adding-new-resources)

## How Can I Contribute?

### 🐛 Reporting Bugs

If you find a bug, please create an issue with:
- A clear, descriptive title
- Steps to reproduce the problem
- Expected vs actual behavior
- Your environment (OS, Python version, PyTorch version)
- Any relevant code snippets or error messages

### 💡 Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:
- A clear, descriptive title
- Detailed description of the proposed enhancement
- Why this enhancement would be useful
- Examples of how it would work

### 📝 Adding Papers, Repositories, or Resources

We welcome additions of:
- New research papers (especially recent CVPR, ICCV, ECCV, NeurIPS papers)
- GitHub repositories with implementations
- Datasets for instance segmentation
- Tutorials and blog posts
- Pretrained models

## Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/Unet-Instance-Segmentation.git
   cd Unet-Instance-Segmentation
   ```

2. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing format and style
   - Test any code examples you add
   - Update documentation as needed

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template

## Submission Guidelines

### Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Your code follows the project's style guidelines
- [ ] You've tested any code examples
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] No merge conflicts exist
- [ ] Links are valid and working
- [ ] Papers include proper citations
- [ ] New sections fit logically into existing structure

### Commit Message Format

Use clear, descriptive commit messages:

```
Add: New paper on transformer-based U-Net
Fix: Broken link in datasets section
Update: Performance metrics for IAUNet
Docs: Add tutorial for mobile deployment
Refactor: Reorganize U-Net variants section
```

## Code Style Guidelines

### Python Code

- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to functions and classes
- Include type hints where appropriate
- Keep functions focused and concise

Example:
```python
def preprocess_image(image: np.ndarray, target_size: Tuple[int, int]) -> torch.Tensor:
    """
    Preprocess image for model input.

    Args:
        image: Input image as numpy array [H, W, C]
        target_size: Target size as (height, width)

    Returns:
        Preprocessed image tensor [C, H, W]
    """
    # Implementation here
    pass
```

### Markdown

- Use proper heading hierarchy (H1 → H2 → H3)
- Include descriptive link text
- Keep tables properly formatted
- Use code blocks with language specification
- Add emojis for visual appeal (but don't overdo it)

## Adding New Resources

### Adding a Paper

Papers should be added to the appropriate section with:

```markdown
| **Paper Title** | Venue Year | Key contribution | [Paper](url) \| [Code](url) |
```

Required information:
- Full paper title
- Conference/Journal and year
- Brief description of key contribution
- Link to paper (preferably arXiv or official proceedings)
- Link to code repository (if available)

### Adding a Repository

Repositories should include:

```markdown
| **Repository Name** | Stars | Description | [Link](url) |
```

Required information:
- Repository name
- Approximate star count (if significant)
- Brief description
- Working link to repository

### Adding a Dataset

Datasets should include:

| Dataset | Type | Size | Application | Link |
|---------|------|------|-------------|------|
| Name | Domain | # images/instances | Use case | [Link](url) |

Required information:
- Dataset name
- Type/domain
- Size information
- Primary application
- Download or information link

### Adding Code Examples

Code examples should:
- Be complete and runnable
- Include necessary imports
- Have clear comments
- Follow Python best practices
- Include docstrings
- Be placed in appropriate `/examples` subdirectory

## Review Process

1. **Initial Review**: Maintainers review PR for basic requirements
2. **Technical Review**: Code is tested and validated
3. **Documentation Review**: Documentation is checked for accuracy
4. **Final Review**: Overall contribution is assessed
5. **Merge**: PR is merged if approved

## Recognition

Contributors will be:
- Listed in the repository's contributors
- Acknowledged in release notes
- Credited in relevant documentation

## Questions?

If you have questions:
- Check existing issues
- Create a new issue with the `question` label
- Reach out to maintainers

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to Awesome U-Net Instance Segmentation!** 🎉
