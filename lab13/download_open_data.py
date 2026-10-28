"""Загрузить небольшой набор реальных фотографий бабочек с индивидуально проверенной CC0-лицензией Wikimedia Commons.

Зависимость: Pillow. Не объявляйте лицензии по результатам поиска: проверяется
imageinfo.extmetadata для КАЖДОГО файла; при изменении лицензии загрузка падает.
Каталог назначения: CV_LORA_DATA либо data/lab13 относительно текущей лабы.
"""
import hashlib
import io
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from PIL import Image

API = 'https://commons.wikimedia.org/w/api.php'
HEADERS = {'User-Agent': 'CVUniversityLab/0.1 (educational research; contact: cv-lab@localhost.invalid)'}
TITLES = [
    '20230130-butterfly.jpg', 'A Beautiful Black Butterfly.jpg',
    'A butterfly eating papaya, Location- Lakhimpur, Assam.jpg',
    'A butterfly on a purple flower.jpg', 'Black and white butterfly near Shirahama.jpg',
    'Butterfly 51.jpg', 'Butterfly Iguazu Falls.jpg', 'Butterfly The Insect.jpg',
    'Butterfly at Jonanjima.jpg', 'Butterfly at Wingham Wildlife Park Kent UK.jpg',
    'Butterfly by the water.jpg', 'Butterfly enjoying with nature.jpg',
    'Butterfly from butterfly garden-lucknow zoo.jpg', 'Butterfly hanging on leaf.jpg',
    'Butterfly in Armenia.jpg', 'Butterfly orange.jpg',
    'Common Jezebel Delias eucharis by kadavoor 3.jpg',
    'Common Pierrot butterfly at Anamangad, Malappuram, Kerala.jpg',
    'Common cerulean butterfly at Sooranad North.jpg',
    'Dark Cerulean Butterfly Female.jpg', 'Elymnias hypermnestra male by kadavoor.jpg',
    'Marbled White Butterfly (Melanargia galathea).jpg',
    'Monarch butterfly on butterfly bush flower.jpg',
    'Red Pierrot Butterfly at Jakkur, Bangalore.jpg',
    'Red admiral butterfly with wings closed.jpg', 'Swallowtail butterfly up close.jpg',
    'White butterfly upside down on leaf.jpg', 'Yellow and black butterfly on ground.jpg',
]


def fetch(url):
    with urlopen(Request(url, headers=HEADERS), timeout=45) as response:
        return response.read()


def download(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    params = {'action': 'query', 'titles': '|'.join('File:' + title for title in TITLES),
              'prop': 'imageinfo', 'iiprop': 'url|extmetadata|size|sha1|mime',
              'iiurlwidth': '640', 'format': 'json', 'formatversion': '2'}
    payload = json.loads(fetch(API + '?' + urlencode(params)))
    pages = {page['title']: page for page in payload['query']['pages']}
    if set(pages) != {'File:' + title for title in TITLES}:
        raise ValueError('Commons изменил названия файлов: проверьте подборку вручную')
    records = []
    for index, title in enumerate(TITLES):
        page = pages['File:' + title]
        info = page['imageinfo'][0]
        metadata = info['extmetadata']
        license_name = metadata['LicenseShortName']['value']
        license_url = metadata['LicenseUrl']['value']
        if license_name != 'CC0' or 'creativecommons.org/publicdomain/zero/1.0/' not in license_url:
            raise ValueError(f'Не подтверждена CC0 для {title}: {license_name} {license_url}')
        if info['mime'] != 'image/jpeg':
            raise ValueError(f'Не JPEG-фотография: {title}')
        filename = f'commons_{index:02d}.jpg'
        path = destination / filename
        if path.exists():
            raw = path.read_bytes()
        else:
            raw = fetch(info['thumburl'])
            # Не фиксируем файл до проверки формата.
            with Image.open(io.BytesIO(raw)) as image:
                image.verify()
            path.write_bytes(raw)
        with Image.open(io.BytesIO(raw)) as image:
            image.convert('RGB').load()
            if min(image.size) < 256:
                raise ValueError(f'Слишком маленькое изображение: {title}: {image.size}')
        records.append({'file_name': filename,
                        'text': 'a photo of sks butterfly, ' + title.removesuffix('.jpg').replace('_', ' ').lower(),
                        'license': 'CC0-1.0', 'license_url': license_url,
                        'source_url': info['descriptionurl'],
                        'download_url': info['thumburl'], 'original_sha1': info['sha1'],
                        'sha256': hashlib.sha256(raw).hexdigest(),
                        'source_title': page['title'],
                        'author_as_reported': metadata.get('Artist', {}).get('value', '')})
        print(f'{index + 1}/{len(TITLES)} {filename}: {title} ({len(raw)} bytes)')
    if len({r['sha256'] for r in records}) != len(records):
        raise ValueError('Повторяющиеся изображения')
    output = destination / 'metadata.jsonl'
    output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    print(f'{len(records)} реальных RGB JPEG, индивидуально подтверждённых Commons CC0; {output}')
    return records


if __name__ == '__main__':
    download(os.environ.get('CV_LORA_DATA', 'data/lab13'))
