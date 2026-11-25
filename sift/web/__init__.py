"""SIFT Gate-1 invited beta web API and operator CLI package.

Hard rules:
- Zero raw market data, ticker symbols, or directional calls ever leave the server.
- All user-facing card/deck/onboarding copy comes from `sift.cards.copy` and `sift.decks.copy`.
- Email PII is isolated in `auth.db` behind opaque `user_id`s.
"""
