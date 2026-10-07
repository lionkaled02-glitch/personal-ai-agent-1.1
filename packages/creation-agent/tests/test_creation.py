from creation_agent import CreationService, ScriptRequest


def test_script_and_storyboard_are_bounded_and_deterministic() -> None:
    service = CreationService()
    request = ScriptRequest(topic="Explain photosynthesis", target_duration_s=30)
    first = service.create_script(request)
    second = service.create_script(request)
    assert first == second
    board = service.storyboard(first, scenes=4)
    assert len(board.scenes) == 4
    assert all(scene.duration_s > 0 for scene in board.scenes)
