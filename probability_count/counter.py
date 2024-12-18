import eval7
from eval7 import Card, HandRange

RANGES = {
    'nit': 'AA, KK, QQ, JJ, AKs',
    'tight': 'AA, KK, QQ, JJ, TT, 99, AKs, AQs, AJs, KQs, AKo, AQo',
    'medium': 'AA, KK, QQ, JJ, TT, 99, 88, 77, AKs, AQs, AJs, ATs, KQs, KJs, QJs, AKo, AQo, AJo, KQo, A5s-A2s',
    'loose': 'AA-22, AKs-A2s, KQs-K2s, QJs-Q2s, JTs-J2s, T9s-T2s, 98s-92s, 87s-82s, 76s-72s, 65s-62s, 54s-52s, 43s-42s, 32s, AKo-A2o, KQo-K2o, QJo-Q2o, JTo-J2o, T9o-T2o, 98o-92o, 87o-82o, 76o-72o, 65o-62o, 54o-52o, 43o-42o, 32o',
    'all': 'AA-22, AKs-A2s, KQs-K2s, QJs-Q2s, JTs-J2s, T9s-T2s, 98s-92s, 87s-82s, 76s-72s, 65s-62s, 54s-52s, 43s-42s, 32s, AKo-A2o, KQo-K2o, QJo-Q2o, JTo-J2o, T9o-T2o, 98o-92o, 87o-82o, 76o-72o, 65o-62o, 54o-52o, 43o-42o, 32o'
}

def calculate_equity(hand_str, board_str, player_type='medium', num_opponents=1, iterations=100000):
    if player_type not in RANGES:
        raise ValueError(f"Unknown player type. Available types: {', '.join(RANGES.keys())}")

    hand = tuple(map(Card, hand_str))
    board = tuple(map(Card, board_str)) if board_str else []

    villain_range = HandRange(RANGES[player_type])

    equity = eval7.py_hand_vs_range_monte_carlo(
        hand,
        villain_range,
        board,
        iterations
    )

    if num_opponents > 1:
        equity = equity ** num_opponents

    return equity


def get_all_equity(hand, board=[], num_opponents=1):
    equities = {}
    for player_type in ['nit', 'tight', 'medium', 'loose', 'all']:
        equities[player_type] = calculate_equity(hand, board, player_type, num_opponents)*100

    return equities


def print_equity_vs_all_types(hand, board=None, num_opponents=1):
    board = board or []

    for player_type in ['nit', 'tight', 'medium', 'loose', 'all']:
        equity = calculate_equity(hand, board, player_type, num_opponents)
        print(f"{player_type.capitalize():8}: {equity:.2%}")


# hand = ["6d", "Qd"]
# board = []
#
# print_equity_vs_all_types(hand, board, num_opponents=1)
