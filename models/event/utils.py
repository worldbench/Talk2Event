import math


def _get_modified_hw_multiple_of(hw, multiple_of):
    assert isinstance(hw, tuple), f'{type(hw)=}, {hw=}'
    assert len(hw) == 2
    assert isinstance(multiple_of, int)
    assert multiple_of >= 1
    if multiple_of == 1:
        return hw
    new_hw = tuple(math.ceil(x / multiple_of) * multiple_of for x in hw)
    return new_hw