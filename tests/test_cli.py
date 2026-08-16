from imrnns.cli import build_parser, main


def test_cli_exposes_public_commands(capsys):
    parser = build_parser()
    help_text = parser.format_help()
    for command in ("info", "download", "cache", "train", "evaluate", "run"):
        assert command in help_text
    assert "reproduce" not in help_text
    assert "convert-checkpoint" not in help_text
    assert main(["info"]) == 0
    output = capsys.readouterr().out
    assert '"version": "0.2.0"' in output
    assert '"objective": "improvement-margin"' in output


def test_cli_training_defaults_match_validated_recipe():
    args = build_parser().parse_args(["train", "--dataset", "scifact", "--encoder", "minilm"])
    assert args.improvement_margin == 0.05
    assert args.num_negatives == 63
    assert args.epochs == 30
    assert args.patience == 7
    assert not hasattr(args, "loss")
    assert not hasattr(args, "modulation_mode")
    assert not hasattr(args, "optimizer")

    cache_args = build_parser().parse_args(["cache", "--dataset", "scifact", "--encoder", "minilm"])
    assert cache_args.num_negatives == 63
    assert not hasattr(cache_args, "negative_method")
