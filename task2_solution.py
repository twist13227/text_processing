PARENTHESISES = r'(\(.\)|{.}|\[.\])*'


def tree(regexp, n):
    if n == 0:
        return regexp.replace('.', '')
    return tree(regexp.replace('.', PARENTHESISES), n-1)


PARENTHESIS_REGEXP = tree(PARENTHESISES, 10)


SENTENCES_REGEXP = r'(?P<sentence>^.*?[\.\?!]|(?<=[\.\?!]\s).*?[\.\?!])'


PERSONS_REGEXP = r'(?P<person>[А-Я][а-я]+\s[А-Я][а-я]+|[А-Я][а-я]+\s[А-Я][а-я]+\s[А-Я][а-я]+)'


NAME = r'<h1.*><a.*href=\"/series/\d*/\">(?P<name>.*)</a></h1>'
EPISODES_COUNT = r'<td.*><b>Эпизоды:</b></td>((.|\n)*)<td.*>(?P<episodes_count>\d+)</td>'
EPISODE_NUMBER = r'(?P<episode_number>(?<=<span style=\"color:#777\">Эпизод )\d*)'
EPISODE_NAME = r'(?P<episode_name>(?<=<h1 class="moviename-big" style=\"font-size:16px;padding:0px;color:#444\"><b>).*?(?=<\/b><\/h1>))'
EPISODE_ORIGINAL_NAME =r'(?P<episode_original_name>(?<=<span class="episodesOriginalName">).*?(?=</span> </td>))'
EPISODE_DATE = r'(?P<episode_date>(?<=align="left" class="news" style="border-bottom:1px dotted #ccc;padding:15px 0px;font-size:12px" valign="bottom" width="20%">).*?(?=</td>))'
SEASON = r'(?P<season>(?<=<h1 class="moviename-big" style="font-size:21px;padding:0px;margin:0px;color:#f60">Сезон )\d*(?=</h1>))'
SEASON_YEAR = r'(?P<season_year>\d{4}(?=, эпизодов:))'
SEASON_EPISODES = r'(?P<season_episodes>(?<=эпизодов: )\d*)'
SERIES_REGEXP = r'|'.join(
    (
        NAME,
        EPISODES_COUNT,
        EPISODE_NUMBER,
        EPISODE_NAME,
        EPISODE_ORIGINAL_NAME,
        EPISODE_DATE,
        SEASON,
        SEASON_YEAR,
        SEASON_EPISODES
    )
)