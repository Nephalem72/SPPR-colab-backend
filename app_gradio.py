from __future__ import annotations

from sppr_colab.config import settings
from sppr_colab.ui import CSS, create_demo


if __name__ == "__main__":
    auth = None
    if settings.ui_username and settings.ui_password:
        auth = (settings.ui_username, settings.ui_password)
    create_demo().queue(default_concurrency_limit=1).launch(
        server_name=settings.ui_host,
        server_port=settings.ui_port,
        share=settings.ui_share,
        auth=auth,
        show_error=True,
        debug=True,
        footer_links=[],
        css=CSS,
    )
