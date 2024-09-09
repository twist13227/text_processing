PASSWORD_REGEXP = r'(?=.*[0-9].*)(?=.*[A-Z].*)(?=.*[a-z].*)(?![a-zA-Z0-9]*([\^\$%@#&\*!\?])(?:[a-zA-Z0-9]|\1)*$)(?!.*(.)\2.*$)(?:[a-zA-Z0-9\^\$%@#&\*!\?]*){8,}'


RGB = r'^rgb\((0|[1][0-9]?[0-9]?|[2][0-9]?[0-5]?|[3-9][0-9]?)(,\s*(0|([1][0-9]?[0-9]?){0,2}|([2][0-9]?[0-5]?){0,2}|([3-9][0-9]?){0,2})){2}\)$|^rgb\((0%|[1-9][0-9]?%|100%)(,\s*(0%|[1-9][0-9]?%|100%)){2}\)$'
HEX = r'(?:^#([0-9]|[a-fA-F]){6}|^#([0-9]|[a-fA-F]){3})$'
HSL = r'^hsl\((0|[1-9][0-9]?|[12][0-9][0-9]|3[0-5][0-9]|360)(,\s*(0%|[1-9][0-9]?%|100%)){2}\)$'
COLOR_REGEXP = '|'.join(
    (
        RGB,
        HEX,
        HSL,
    )
)


VARIABLE = r'(?P<variable>\b[a-zA-Z_][a-zA-Z_0-9]*\b)'
NUMBER = r'(?P<number>\b[0-9]+(?:\.[0-9]*)?|\.[0-9]+\b)'
CONSTANT = r'(?P<constant>\b(?:pi|e|sqrt2|ln2|ln10)\b)'
FUNCTION = r'(?P<function>\b(?:sin|cos|tg|ctg|tan|cot|sinh|cosh|th|cth|tanh|coth|ln|lg|log|exp|sqrt|cbrt|abs|sign)\b)'
OPERATOR = r'(?P<operator>[\^*/\-+])'
LEFT_PARENTHESIS = r'(?P<left_parenthesis>\()'
RIGHT_PARENTHESIS = r'(?P<right_parenthesis>\))'
EXPRESSION_REGEXP = r'|'.join(
    (
        FUNCTION,
        CONSTANT,
        VARIABLE,
        NUMBER,
        OPERATOR,
        LEFT_PARENTHESIS,
        RIGHT_PARENTHESIS,
    )
)


DAY = r'(?:0*[1-9]|0*[1-2][0-9]|0*3[0-1])'
NUMB_MONTH = r'(?:0*[1-9]|1[0-2])'
RUS_MONTH = r'(?:' + r'янв(аря)?|фев(раля)?|мар(та)?|апр(еля)?|ма(й|я)?|июн(я)?|июля(я)?|авг(уста)?|сен(тября)?|окт(ября)?|ноя(бря)?|дек(абря)?' + r')'
ENG_MONTH = r'(?:' + r'Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?' + r')'
YEAR = r'(?:[0-9]+)'
DATES_REGEXP = '|'.join(
    (
        DAY + r'\.' + NUMB_MONTH + r'\.' + YEAR,
        DAY + r'/' + NUMB_MONTH + r'/' + YEAR,
        DAY + r'-' + NUMB_MONTH + r'-' + YEAR,
        YEAR + r'\.' + NUMB_MONTH + r'\.' + DAY,
        YEAR + r'/' + NUMB_MONTH + r'/' + DAY,
        YEAR + r'-' + NUMB_MONTH + r'-' + DAY,
        DAY + r'\s*' + RUS_MONTH + r'\s*' + YEAR,
        ENG_MONTH + r'\s*' + DAY + r'\s*,\s*' + YEAR,
        YEAR + r'\s*,\s*' + ENG_MONTH + r'\s*' + DAY,
    )
)
