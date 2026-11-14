"""What the trainer hands to ``notify`` when a run is over."""

from dataclasses import dataclass

__all__ = ["Notice"]


@dataclass(frozen=True, slots=True)
class Notice:
    """
    How a run ended, written for someone who was not there to see it.

    Two parts, because every way of reaching a person has two places to put
    words — a mail has a subject and a body, a push notification a title and a
    message, a chat post a first line and the rest. ``str(notice)`` is the two
    together, for anything that takes a single string.

    Attributes:
        title: One line that is enough on its own in a list of subjects: which
            run, how it ended, and the best value or the error it ended on, when
            there is one. The trainer's own words in it are ASCII, so it goes
            into a mail subject or an HTTP header as it stands — unless the run
            directory, the monitored metric or its phase is named outside ASCII,
            in which case a header needs it encoded.
        body: What you would otherwise have to walk to the machine to find out —
            the result, the error if there was one, when and where it ran, and
            where its files are. Plain text, a fact or two to a line.
    """

    title: str
    body: str

    def __str__(self) -> str:
        return f"{self.title}\n\n{self.body}"
