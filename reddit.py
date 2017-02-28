#!/usr/bin/env python3


"""\
A script to download current articles from a Reddit.

It expects an INI file with the login credentials for this script. For more
information about the INI file and its keys and contents, see
https://praw.readthedocs.io/en/latest/getting_started/configuration/prawini.html

By default the script expects you to name this file "reddit.ini", but you can
use the --config command-line option to specify otherwise.

"""


USER_AGENT = 'python:com.ericrochester.insurance:v0.0.0 (by /u/erochest)'


from collections import namedtuple
import csv
import datetime

import click
from click_datetime import Datetime
import praw
from praw.models import MoreComments


Posting = namedtuple('Posting', [
    'id',
    'type',       # either submission or comment
    'parent_id',  # for submission, the subreddit id,
                  # for comment, the submission id
    'permalink',
    'redditor',
    'title',
    'created_at',
    'score',
    'downs',
    'ups',
    'text',
    ])


def submission_posting(submission):
    """Pull the data out of the submission."""
    return Posting(
        submission.id,
        'submission',
        submission.subreddit_id,
        submission.permalink,
        submission.author.name,
        submission.title,
        datetime.datetime.utcfromtimestamp(submission.created_utc),
        submission.score,
        submission.downs,
        submission.ups,
        submission.selftext,
        )


def comment_posting(comment):
    """Pull the data out of the comment."""
    return Posting(
        comment.id,
        'comment',
        comment.parent_id,
        None,
        comment.author.name,
        '',
        datetime.datetime.utcfromtimestamp(comment.created_utc),
        comment.score,
        comment.downs,
        comment.ups,
        comment.body,
        )


@click.command(help=__doc__)
@click.option('-c', '--config-file', default='praw.ini',
              type=click.Path(exists=True, dir_okay=False),
              help='The INI file for logging into Reddit and using its API. '
                   'Default is praw.ini.')
@click.option('-s', '--subreddit-name', help='The subreddit to pull from.')
@click.option('-b', '--bot-name', default='insurance',
              help='The name of the bot in the config file. '
                   'Default is "insurance".',)
@click.option('-f', '--from-date', type=Datetime(format='%m/%d/%Y'),
              help='The starting date to pull submissions and comments from. '
                   'In the format MM/DD/YYYY. (E.g., 02/27/2017).')
@click.option('-t', '--to-date', type=Datetime(format='%m/%d/%Y'),
              help='The final date to pull submissions and comments to. '
                   'In the format MM/DD/YYY.')
@click.option('-o', '--output', help='The output CSV file to create.',
              type=click.Path(dir_okay=False))
def main(config_file, subreddit_name, bot_name, from_date, to_date, output):
    """main"""
    reddit = praw.Reddit(bot_name, user_agent=USER_AGENT)
    subreddit = reddit.subreddit(subreddit_name)
    print(subreddit.title)
    print(subreddit.description)

    with open(output, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(Posting._fields)

        for submission in subreddit.submissions(from_date.timestamp(),
                                                to_date.timestamp()):
            writer.writerow(submission_posting(submission))
            submission.comments.replace_more(limit=0)
            writer.writerows(
                comment_posting(comment)
                for comment in submission.comments.list()
                )


if __name__ == '__main__':
    main()
