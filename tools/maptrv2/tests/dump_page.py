"""Save the real browse page HTML (two tiles, two tags) for the jsdom test."""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, 'test_viewer_views_tags.py')).read()
exec(src.split('# ---------------------------------------------------------- projection maths')[0])
app.post('/tag', json={'uid': UID, 'tag': 'corrupted', 'on': 1})
app.post('/tag', json={'uid': UID, 'tag': 'blurry-lidar', 'on': 1})
h = app.get('/').data.decode()
open(os.environ.get('PAGE_OUT', '/tmpx/page.html'), 'w').write(h)
print('wrote page.html', len(h), 'chars;',
      h.count('class="tagbox"'), 'tagboxes,', h.count('class="tags"'), 'rows')
