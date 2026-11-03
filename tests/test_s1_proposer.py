from kompression.s1.proposer import (
    RandomPolicy,
    SoftmaxBandit,
    compression_credit,
    learnability_reward,
)


def test_learnability_reward_matches_absolute_zero():
    assert learnability_reward(0.0) == 0.0
    assert learnability_reward(1.0) == 0.0
    assert learnability_reward(0.25) == 0.75


def test_compression_credit_is_signed_and_proportional():
    c = compression_credit(-90.0, {"a": 2, "b": 1})
    assert c == {"a": -60.0, "b": -30.0}
    assert compression_credit(10.0, {}) == {}


def test_bandit_explores_all_then_prefers_best():
    arms = ("a", "b", "c")
    b = SoftmaxBandit(arms, temperature=0.2, seed=1)
    seen = set()
    for _ in range(3):
        (arm,) = b.select()
        seen.add(arm)
        b.update(arm, {"a": 1.0, "b": 0.0, "c": 0.0}[arm])
    assert seen == set(arms)
    picks = [b.select()[0] for _ in range(200)]
    assert picks.count("a") > 150


def test_random_policy_covers_arms():
    p = RandomPolicy(("a", "b"), seed=0)
    assert set(p.select(50)) == {"a", "b"}
