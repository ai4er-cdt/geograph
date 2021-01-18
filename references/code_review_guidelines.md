# Code review guidelines

<details><summary>Table of contents</summary>
<p>

1. Why code reviews
2. Reviewee side
3. Pull requests
4. Commit messages
5. Reviewer side
6. Best practices
7. Reviewer checklist

</p>
</details>

## 1. Why code review and coding guidelines?

![XKCD Code Quality](https://imgs.xkcd.com/comics/code_quality.png)

[Source: XKCD 'Code Quality'](https://xkcd.com/1513/)

![Best practice sharing](https://phauer.com/blog/2018/10-code-review-guidelines/author-best-practices-experiences-eo.svg 'caption')

## 2. Organising code

### 2.1 GitHub branches and pull requests

![GitHub Branches and PR](https://docs.github.com/assets/images/help/branches/pr-retargeting-diagram2.png)

[GitHub code review video](https://www.youtube.com/watch?v=HW0RPaJqm4g)

https://www.youtube.com/watch?v=gJDtC_tp5w4

### 2.2 Branches

The name of the branch should reflect its purpose. To keep ourselves organised, let us follow these conventions for branch names.

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

## 3. Writing code

### 3.1 Code style guidelines

Style guides are immensely useful to produce readable code. Sticking to these standards will make it much easier to work with others. Adhering to style guidelines will also make the code review process quicker and more fun.

Style guidelines are entire topic of themselves. To avoid clutter, we will not cover them in depth here. Instead, please refer to the google styleguide for python [here](https://google.github.io/styleguide/pyguide.html). As we will mostly use Python in this project, this should be enough for our purposes. If in doubt, simply check out the styleguide for the best practice.

If you're note yet familiar with styleguides, please give it a read. For a start, particularly the conventions on white space ([3.6](https://google.github.io/styleguide/pyguide.html#3163-file-naming)), comments and docstrings ([3.8](https://google.github.io/styleguide/pyguide.html#3163-file-naming)), file naming ([3.16.3](https://google.github.io/styleguide/pyguide.html#3163-file-naming)) and type annotations ([3.19](https://google.github.io/styleguide/pyguide.html#3163-file-naming) are relevant. Don't worry if it might sound overwhelming at the start. Simply try your best when you write the code and we will help each other with the rest in the code reviews.

TODO: Simon will set up an auto-formatter which mostly implements the style guide directly. Resources: [yapf](https://github.com/google/yapf/) and [pylintrc](https://google.github.io/styleguide/pylintrc).

### 3.2 Automatic formatting and linting

### 3.3 Preparing code for review

### 3.4 Finding reviewers

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
- Try to respond to every comment, even if it's only an "ok" or "thank you".
- Merge once you feel confident in the code and its impact on the project.
- Final editorial control rests with you, the pull request author.

A good example response to style comments:

    > Whoops. Good catch, thanks. Fixed in a4994ec.

## 4. Reviewing Code

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

Mostly taken from [here](https://github.com/thoughtbot/guides/tree/main/code-review).
and [here](https://www.codeproject.com/Articles/524235/Codeplusreviewplusguidelines).
