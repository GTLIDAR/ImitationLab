import io

import pytest

from imitationlab.data_tools.common.omniverse_eula import require_eula_answerable


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.mark.parametrize("value", ["YES", "yes", "Y"])
def test_accepted_eula_passes_without_a_terminal(value):
    require_eula_answerable({"OMNI_KIT_ACCEPT_EULA": value}, io.StringIO())


def test_terminal_can_answer_the_prompt():
    require_eula_answerable({}, _Terminal())


@pytest.mark.parametrize("environ", [{}, {"OMNI_KIT_ACCEPT_EULA": "NO"}])
def test_unattended_run_explains_how_to_accept(environ):
    with pytest.raises(SystemExit, match="OMNI_KIT_ACCEPT_EULA=YES"):
        require_eula_answerable(environ, io.StringIO())
