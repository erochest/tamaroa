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


import click
import praw


@click.command(help=__doc__)
@click.option('-c', '--config-file', default='praw.ini',
              type=click.Path(exists=True, dir_okay=False),
              help='The INI file for logging into Reddit and using its API.')
@click.option('-s', '--subreddit-name', help='The subreddit to pull from.')
@click.option('-b', '--bot-name', help='The name of the bot in the config file.',
              default='insurance')
def main(config_file, subreddit_name, bot_name):
    """main"""
    reddit = praw.Reddit(bot_name, user_agent=USER_AGENT)
    subreddit = reddit.subreddit(subreddit_name)
    print(subreddit.title)
    print(subreddit.description)


if __name__ == '__main__':
    main()
