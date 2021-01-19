# Code review guidelines

<details><summary>Table of contents</summary>
<p>

- [Code review guidelines](#code-review-guidelines)
  - [1. Why code review and coding guidelines?](#1-why-code-review-and-coding-guidelines)
  - [2. Organising code](#2-organising-code)
    - [2.1 GitHub](#21-github)
    - [2.2 Branches](#22-branches)
    - [2.3 Commit messages](#23-commit-messages)
  - [3. Writing code](#3-writing-code)
    - [3.1 Code style guidelines](#31-code-style-guidelines)
    - [3.2 Automatic formatting and linting](#32-automatic-formatting-and-linting)
    - [3.3 Preparing code for review](#33-preparing-code-for-review)
    - [3.4 Assigning reviewers](#34-assigning-reviewers)
    - [3.5 Checklist (Reviewee)](#35-checklist-reviewee)
  - [4. Reviewing Code](#4-reviewing-code)
    - [4.1 Checklist (Reviewer)](#41-checklist-reviewer)
  - [5. Credit](#5-credit)

</p>
</details>

## 1. Why code review and coding guidelines?

![XKCD Code Quality](https://imgs.xkcd.com/comics/code_quality.png)

[Source: XKCD 'Code Quality'](https://xkcd.com/1513/)

We perform code reviews (CRs) in order to improve code quality and benefit from positive effects on team culture. They allow us to share knowledge and best practices, keep our project consistent and motivate each other to write legible code. Additionally, they will allow us to catch accidental or strucutral errors in our code and limit the accumulation of [technical debt](https://en.wikipedia.org/wiki/Technical_debt).

Remember that code reviews should be a fun and rewarding procedure. So please speak up whenever you have ideas on how to improve the process or if any of the guidelines do not work for you.

Below are a set of guidelines to structure our team coding and our code reviews. Enjoy the read! 😄

## 2. Organising code

### 2.1 GitHub

We will use GitHub as a means to organise and version control our code. One of the many perks of GitHub for team coding, is the [code review feature for pull requets](https://www.youtube.com/watch?v=HW0RPaJqm4g), which we will use for our code reviews.

If you are not yet familiar with git or github, please ask your other team members for help.

### 2.2 Branches

Branches are used to develop features isolated from each other. The **master** or **main** branch is the "default" branch when you create a repository. We use other branches for development and **merge** them back to the master branch upon completion. This keeps the master branch clean and up-to-date.

The process of merging a branch into another branch is called a **pull request (PR)**. Upon opening a pull request on GitHub, a **continuous integration (CI)** can be triggered to check the code for compliance with certain guidelines. Reviewers can then comment the code on the pull request, provide feedback and ultimately sign it off as "ready to merge". Here's a simplified, visual depiction of the process from githubs official documentation.

![GitHub Branches and PR](https://docs.github.com/assets/images/help/branches/pr-retargeting-diagram2.png)

The name of a branch should reflect its purpose. To keep ourselves organised, let us follow these conventions for branch names.

A new branch should follow this naming pattern `purpose/short-explanation` (all lower case, hyphen separated).

In the above, `purpose` should be one of:

- **feature**: A new feature. For example, a new machine learning model class or preprocessing routine. Example:
  - feature/sentinel1-api
  - feature/cloud-removal-preprocessing
- **exp**: Any experimental idea or feature. This is meant for data explorations and playing around with ideas that will not neccessarily be integrated in the master branch. Example:
  - exp/sentinel1-data-exploration
  - exp/svm-data-exploration
- **fix**: Any bug fixes to existing code. Example:
  - fix/sentinel-data-loading
- **doc**: Documentation only changes. Example:
  - doc/code-review-guidelines
- **refactor**: Any changes that neither fix a bug nor add a new feature. This is meant for simple reformatting of the code base or simplifying existing code without changing the functionality. Example:
  - refactor/remove-unused-files

### 2.3 Commit messages

This section follows [this widely quoted advice for commit messages](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html).
Use the same `purpose` keywords for the branches above for the first word of your commit (i.e. feature, exp, fix, ...).

Here's an example of a bad commit message:

```
# BAD:
make preprocessing work again
```

And here's a good commit message:

```
# GOOD:
fix: add numpy dependency to fix sentinel preprocessing
```

![Commit image](https://datasciencecampus.github.io/coding-standards/rsrc/img/git_commit.png)  
[Source](https://datasciencecampus.github.io/coding-standards/version-control.html#branch-names)

If you have a more complicated change, you can follow this general model:

```
purpose: lowercase, short (50 chars or less) summary

More detailed explanatory text, if necessary. In some contexts, the first line is treated as the subject of an email and the rest of the text as the body.  The blank line separating the summary from the body is critical (unless you omit the body entirely).

Write your commit message in the imperative: "Fix bug" and not "Fixed bug" or "Fixes bug."  This convention matches up with commit messages generated by commands like git merge and git revert.

Further paragraphs come after blank lines.

- Bullet points are okay, too

- Typically a hyphen or asterisk is used for the bullet, followed by a
  single space, with blank lines in between, but conventions vary here

- Use a hanging indent
```

## 3. Writing code

### 3.1 Code style guidelines

Style guides are immensely useful to produce readable code. Sticking to these standards will make it much easier to work with others. Adhering to style guidelines will also make the code review process quicker and more fun.

Style guidelines are entire topic of themselves. To avoid clutter, we will not cover them in depth here. Instead, please refer to the google styleguide for python [here](https://google.github.io/styleguide/pyguide.html). As we will mostly use Python in this project, this should be enough for our purposes. If in doubt, simply check out the styleguide for the best practice.

If you're note yet familiar with styleguides, please give it a read. For a start, particularly the conventions on white space ([3.6](https://google.github.io/styleguide/pyguide.html#3163-file-naming)), comments and docstrings ([3.8](https://google.github.io/styleguide/pyguide.html#3163-file-naming)), file naming ([3.16.3](https://google.github.io/styleguide/pyguide.html#3163-file-naming)) and type annotations ([3.19](https://google.github.io/styleguide/pyguide.html#3163-file-naming) are relevant. Don't worry if it might sound overwhelming at the start. Simply try your best when you write the code and we will help each other with the rest in the code reviews.

### 3.2 Automatic formatting and linting

Automatic formatters and linters will greatly simplify the adherence to coding guidelines.

For our project we will use [black](https://black.readthedocs.io/en/stable/) as a python formatter and
[pylint](http://pylint.pycqa.org/en/latest/) as linter. The `pylintrc` file in our project directory contains the rules of the styleguide above and checks the code against those rules. The pre-commit hook will make sure that this check happens on all newly staged files.

To set up the system, perform the following commands in the project directory:

```bash
pip install black pylint pre-commit
make precommit
# If you use VSCode, this will set up your editor by automatically activating black and pylint
make vscode_pro
```

### 3.3 Preparing code for review

This section is mostly taken from [here](https://medium.com/palantir/code-review-best-practices-19e02780015f).
It is the author's responsibility to submit CRs that are easy to review in order not to waste reviewers' time and motivation:

-**Scope and size**. Changes should have a narrow, well-defined, self-contained scope that they cover exhaustively. For example, a change may implement a new feature or fix a bug. Shorter changes are preferred over longer ones. If a CR makes substantive changes to more than ~5 files, or took longer than 1–2 days to write, or would take more than 20 minutes to review, consider splitting it into multiple self-contained CRs. For example, a developer can submit one change that defines the API for a new feature in terms of interfaces and documentation, and a second change that adds implementations for those interfaces.

- Only submit **complete, self-reviewed, and self-tested CRs**. In order to save reviewers' time, test the submitted changes (i.e., run the test suite) and make sure they pass all builds as well as all tests and code quality checks, both locally and on the CI servers, before assigning reviewers.

- **Refactoring** changes should not alter behavior; conversely, a behavior-changing changes should avoid refactoring and code formatting changes. There are multiple good reasons for this:
  - Refactoring changes often touch many lines and files and will consequently be reviewed with less attention. Unintended behavior changes can leak into the code base without anyone noticing.
  - Large refactoring changes break cherry-picking, rebasing, and other source control magic. It is very onerous to undo a behavior change that was introduced as part of a repository-wide refactoring commit.
  - Expensive human review time should be spent on the program logic rather than style, syntax, or formatting debates. We prefer settling those with automated tooling like Checkstyle, TSLint, Baseline, Prettier, etc.

### 3.4 Assigning reviewers

Make sure you do not request maximally 2 reviewers. Reviews between more than three parties are often unproductive.

Also make sure that you request reviews from differnt people over the course of the project to maximise your learning.

### 3.5 Checklist (Reviewee)

Before submitting the review, ask yourself ([print version](http://insidecoding.files.wordpress.com/2013/01/codereview_checklistfordevelopers.docx)):

- My code compiles
- My code has been developer-tested and includes unit tests where appropriate
- My code is tidy (indentation, line length, no commented-out code, no spelling mistakes, automatically formated and linted)
- My code passes CI (continuous integration)
- I have considered proper use of exceptions
- I have made appropriate use of logging
- I have eliminated unused imports
- The code follows the Coding Standards
- Are there any leftover stubs or test routines in the code?
- Are there any hardcoded, development only things still in the code?
- If relevant, was performance considered?
- Can any code be replaced by calls to external reusable components or library functions?

As you get feedback:

- Create a checklist of common feedback you get and check your code against it before you submit.
- Be humble. No matter how good you are, you can still learn and improve.
- Be grateful for the reviewer's suggestions. ("Good call. I'll make that
  change.")
- You are not your code. Programming is just a skill. It improves with training - and this never stops. Don't connect your self-worth with the code you are writing.
- Learn from your peers. Code reviews are a valuable source of best practices and experiences.
- Code reviews are a discussion, not a dictation. It's fine to disagree, but you have to elaborate your reservations politely and be willing to make compromises.
- Explain why the code exists. ("It's like that because of these reasons. Would
  it be more clear if I rename this class/file/method/variable?")
- Link to the code review from a GitHub issue if present. ("Ready for review:
  https://github.com/organization/project/pull/1")
- Try to respond to every comment, even if it's only an "ok" or "thank you" or "done".
- Merge once you feel confident in the code and its impact on the project.
- Final editorial control rests with you, the pull request author.

A good example response to style comments:

    > Whoops. Good catch, thanks. Fixed in a4994ec.

## 4. Reviewing Code

A code review is a synchronization point among different team members and thus has the potential to block progress. Consequently, code reviews need to be prompt (on the order of hours, not days), and team members and leads need to be aware of the time commitment and prioritize review time accordingly. If you don't think you can complete a review in time, please let the committer know right away so they can find someone else.

A review should be thorough enough that the reviewer could explain the change at a reasonable level of detail to another developer. This ensures that the details of the code base are known to more than a single person.

As a reviewer, it is your responsibility to enforce coding standards and keep the quality bar up. Reviewing code is more of an art than a science. The only way to learn it is to do it; an experienced reviewer should consider putting other less experienced reviewers on their changes and have them do a review first.

### 4.1 Checklist (Reviewer)

Understand what the purpose of the branch under review is (fixes a bug, adds new machine learning model, refactors the existing code). Then:

- Communicate ideas and suggestions effectively:

  - Praise, praise, praise.
  - Use keys such as `nit` (nitpicky), `suggestion` to formulate optional feedback.
  - Offer alternative implementations, but assume the author already considered
    them. ("What do you think about using a custom validator here?")
  - Use I-messages:
    > Right: "It's hard for me to grasp what's going on in this code."  
    > Wrong: "You are writing cryptic code."
  - Talk about the code, not the coder.
    > Right: "This code is requesting the service multiple times, which is inefficient."  
    > Wrong: "Youre requesting the service multiple times, which is inefficient."
  - Ask questions instead of making statements (avoid 'why' questions though).
    > Right: "What do you think about the name 'userId' for this variable**?**"  
    > Wrong: "This variable should have the name 'userId'."
  - Mind the OIR-Rule of giving feedback
    - Observation - "This method has 100 lines."
    - Impact - "This makes it hard for me to grasp the essential logic of this method."
    - Request - "I suggest extracting the low-level-details into subroutines and give them expressive names."
  - Before giving feedback, ask yourself:
    - Is it true? (opinion != truth)
    - Is it necessary? (avoid nagging, unnecessary comments and out-of-scope work)
    - Is it kind? (no shaming)
  - It's fine to say: Everything is good!
  - Don't forget to praise.
  - If discussions turn too philosophical or academic, turn to a higher bandwith medium such as video calls or discuss in person.
  - Sign off on the pull request with a "👍 Ready to merge" comment once you think all feedback has been properly addressed. Clearly flag needed changes otherwise.

- Read attentively:
  - Code reviews should be fun. Don't rush it, but also don't spend much more than 20 - 30 min reviewing. Reviews should be short enough to allow you to go through quickly.
  - Code reviews should be a learing experience. Try to understand what the author does and how you can learn from it, or provide a suggestion if you have an idea how to simplify it.
  - Remember that you are here to provide feedback, not to be a gatekeeper.
  - Check for ([print version](http://insidecoding.files.wordpress.com/2013/01/codereview_checklistforreviewers.docx))
    - Comments are comprehensible and add something to the maintainability of the code
    - Comments are neither too numerous nor verbose
    - Type annotations have been given where possible and were used appropriately
    - Exceptions have been used appropriately
    - Repetitive code has been factored out
    - Frameworks have been used appropriately
    - Functions and methods fulfill one task only.
    - If relevant, Unit tests are present and correct.
    - Common errors have been checked for
    - If relevant, Performance was considered
    - The functionality fits the current design/architecture
    - The code complies to coding standards
    - Logging used appropriately (proper logging level and details)

Reviewers should comment on missed style guidelines. Example comment:

    > Order resourceful routes alphabetically by name.

## 5. Credit

Advice mostly taken from [here](https://github.com/thoughtbot/guides/tree/main/code-review), [here](https://medium.com/palantir/code-review-best-practices-19e02780015f)
and [here](https://www.codeproject.com/Articles/524235/Codeplusreviewplusguidelines).
