"""
FinTrust Digital Bank — Customer Support Assistant Prototype
Grounded strictly on FinTrust Financial Knowledge Base.
"""

from typing import Dict, Any

class FinTrustSupportAssistant:
    """Customer-support assistant grounded exclusively on approved policies."""

    SENSITIVE_CREDENTIALS = {'pin', 'password', 'passcode', 'otp', 'authentication code', 'cvv'}

    APPROVED_KNOWLEDGE_BASE = {
        "account_access": (
            "If you cannot sign in, please use the account recovery option in the FinTrust "
            "digital banking interface and follow the verification steps presented there. "
            "If recovery is unsuccessful, your case should be escalated to FinTrust customer support."
        ),
        "failed_transaction": (
            "For a failed transaction, please first confirm that the transaction status is marked "
            "'failed' in your transaction history and avoid repeatedly submitting the same transaction "
            "until the cause is understood."
        ),
        "reversed_transaction": (
            "For a reversed transaction, please review your transaction history and allow the "
            "reversal process to complete before attempting another payment."
        ),
        "unrecognised_transaction": (
            "For a transaction you do not recognise, this must be treated as a security concern "
            "and escalated through the approved support process."
        ),
        "transfers": (
            "Please verify recipient details before confirming any transfer. If a transfer appears "
            "pending, check the transaction status in your app and use the approved support channel "
            "if the issue remains unresolved."
        ),
        "card_compromise": (
            "For card-payment issues, verify the transaction status and merchant details in your transaction history. "
            "If you believe your card or payment credential has been compromised, please use the approved "
            "security/support process immediately and avoid sharing sensitive credentials."
        )
    }

    def generate_response(self, user_query: str) -> str:
        query_clean = user_query.strip().lower()

        # Safety Guardrail: Intercept Sensitive Credentials
        for cred in self.SENSITIVE_CREDENTIALS:
            if cred in query_clean:
                return (
                    "SECURITY ALERT: Customers should never share passwords, PINs, one-time passcodes, "
                    "or authentication codes with another person or with this assistant. "
                    "For security concerns, please use the approved FinTrust security escalation process."
                )

        # Safety Guardrail: Intercept Personal Financial Advice Requests
        advice_triggers = {'invest', 'crypto', 'which stock', 'financial advice', 'loan decision'}
        if any(trigger in query_clean for trigger in advice_triggers):
            return (
                "POLICY NOTICE: This assistant does not provide personalised financial, investment, "
                "lending, or legal advice. Please speak with an authorized banking representative."
            )

        # Strict Knowledge Base Retrieval
        if any(term in query_clean for term in ['cannot sign in', 'login', 'locked', 'recover', 'access']):
            return self.APPROVED_KNOWLEDGE_BASE["account_access"]

        if any(term in query_clean for term in ['failed transaction', 'payment failed', 'failed payment']):
            return self.APPROVED_KNOWLEDGE_BASE["failed_transaction"]

        if any(term in query_clean for term in ['reversed', 'reversal']):
            return self.APPROVED_KNOWLEDGE_BASE["reversed_transaction"]

        if any(term in query_clean for term in ['unrecognised', 'unrecognized', 'did not make', 'fraud']):
            return self.APPROVED_KNOWLEDGE_BASE["unrecognised_transaction"]

        if any(term in query_clean for term in ['transfer', 'pending transfer', 'send money']):
            return self.APPROVED_KNOWLEDGE_BASE["transfers"]

        if any(term in query_clean for term in ['card', 'compromised', 'stolen card', 'lost card']):
            return self.APPROVED_KNOWLEDGE_BASE["card_compromise"]

        # Handling Unsupported Inquiries (Fees, Limits, Unlisted Topics)
        if any(term in query_clean for term in ['fee', 'limit', 'charge', 'cost', 'maximum transfer']):
            return (
                "The training knowledge base does not define a complete fee schedule or transaction-limit table. "
                "This information is not available in the approved knowledge base; we recommend escalating your request."
            )

        # Default Escalation Path
        return (
            "This information is not available in the approved knowledge base. "
            "Please escalate this matter to FinTrust customer support through the approved support process."
        )


if __name__ == "__main__":
    bot = FinTrustSupportAssistant()
    print("Grounding Test: Daily Limit Inquiry")
    print("Response:", bot.generate_response("What is the daily transfer limit?"))
    print("\nSafety Test: User Provides PIN")
    print("Response:", bot.generate_response("My PIN is 1234, can you unlock my card?"))
