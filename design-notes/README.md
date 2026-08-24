# design-notes

One file per working session, named `YYYY-MM-DD-topic.md`, recording what was **found** and what
was **decided** — including the things that were tried and did not work, with the numbers that
killed them.

The point is not a changelog: git already has one, and commit messages already carry the
reasoning for the change they belong to. What git does not hold well is the shape of a whole
session — that four plausible ideas were measured and all four failed for one underlying reason,
or that a parameter was calibrated at the wrong scale and every conclusion drawn from it had to
be redone. Those are the things that get rediscovered and re-attempted months later.

So each note should make it cheap to answer: *has this been tried, and what happened?*

Conventions that have earned their place:

- **Numbers, not adjectives.** "The distances concentrate" is not useful; "`innodb`, whose list is
  perfect, has the farthest first neighbour of every token examined (0.501)" is.
- **Record the failures at the same resolution as the successes.** A dead end documented with its
  measurement is a result. A dead end documented as "didn't work" will be walked again.
- **Name the mechanism, not just the outcome.** Why a thing failed is what transfers to the next
  idea; that a thing failed does not.
