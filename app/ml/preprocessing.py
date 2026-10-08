import re 

def clean_and_sanitize_text(raw_text: str) -> str:
    """Очищает текст от лишних пробелов, переносов строк и маскирует персональные данные.

    Функция находит в тексте все email-адреса и российские номера телефонов,
    заменяя их на заглушки [EMAIL] и [PHONE]. Также удаляются любые
    дублирующиеся пробельные символы, включая переносы строк.

    Args:
        raw_text (str): Исходная необработанная строка текста.

    Returns:
        str: Очищенная строка без персональных данных и лишних пробелов.
        Если на вход подана пустая строка, вернется пустая строка.

    Examples:
        >>> clean_and_sanitize_text("Привет! Напиши мне на test@mail.ru  \n или позвони  8-999-123-45-67")
        'Привет! Напиши мне на [EMAIL] или позвони [PHONE]'
    """
    if not raw_text:
        return ""

    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9_.+-]+\.[a-zA-Z0-9-.]+'
    text = re.sub(email_pattern, '[EMAIL]', raw_text)

    phone_pattern = r'(?:\+7|8)[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}'
    text = re.sub(phone_pattern, '[PHONE]', text)

    cleaned_text = " ".join(text.split())

    return cleaned_text