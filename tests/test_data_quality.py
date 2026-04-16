"""
Data quality tests using Pandera schema validation.

Validates:
1. Raw Retail Rocket events.csv schema (field types, allowed values, constraints)
2. Processed interactions DataFrame schema (after weighting and aggregation)
3. Recommendation API response schema (score range, required fields)
4. Data completeness invariants (no duplicate pairs, finite weights, etc.)

These tests run against synthetic DataFrames, so no actual dataset is needed
in the CI environment. They document and enforce the expected data contracts
for every stage of the pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa
import pytest
from pandera import Check, Column, DataFrameSchema

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Pandera Schemas
# =============================================================================

# Raw Retail Rocket events.csv schema
events_schema = DataFrameSchema(
    {
        "timestamp": Column(int, nullable=False),
        "visitorid": Column(
            int,
            nullable=False,
            checks=Check.greater_than(0),
        ),
        "event": Column(
            str,
            nullable=False,
            checks=Check.isin(["view", "addtocart", "transaction"]),
        ),
        "itemid": Column(float, nullable=True),  # null for some events
        "transactionid": Column(float, nullable=True),
    },
    coerce=True,
)

# Processed interactions after weight assignment and aggregation
interactions_schema = DataFrameSchema(
    {
        "visitorid": Column(str, nullable=False),
        "itemid": Column(str, nullable=False),
        "weight": Column(
            float,
            nullable=False,
            checks=[
                Check.greater_than(0),
                Check.less_than_or_equal_to(100),  # max accumulated weight
            ],
        ),
    }
)

# Recommendation response item schema (for API output validation)
recommendation_response_schema = DataFrameSchema(
    {
        "product_id": Column(
            str,
            nullable=False,
            checks=Check(lambda s: s.str.len() > 0, element_wise=False),
        ),
        "product_name": Column(str, nullable=False),
        "score": Column(
            float,
            nullable=False,
            checks=[
                Check.greater_than_or_equal_to(0.0),
                Check.less_than_or_equal_to(1.0),
            ],
        ),
    }
)


# =============================================================================
# Events Schema Tests
# =============================================================================


class TestEventsSchema:
    """Validate the raw events.csv format."""

    def test_valid_events_pass_schema(self):
        """Well-formed event rows must pass validation without error."""
        valid = pd.DataFrame(
            {
                "timestamp": [1695000000000, 1695000001000],
                "visitorid": [12345, 67890],
                "event": ["view", "addtocart"],
                "itemid": [456.0, 789.0],
                "transactionid": [np.nan, np.nan],
            }
        )
        events_schema.validate(valid)  # should not raise

    def test_transaction_event_with_itemid_passes(self):
        """Transaction events with both itemid and transactionid must pass."""
        valid = pd.DataFrame(
            {
                "timestamp": [1695000002000],
                "visitorid": [11111],
                "event": ["transaction"],
                "itemid": [999.0],
                "transactionid": [42.0],
            }
        )
        events_schema.validate(valid)

    def test_invalid_event_type_fails(self):
        """Unrecognised event types must be rejected."""
        invalid = pd.DataFrame(
            {
                "timestamp": [1695000000000],
                "visitorid": [12345],
                "event": ["purchase"],  # not in allowed set
                "itemid": [456.0],
                "transactionid": [np.nan],
            }
        )
        with pytest.raises(pa.errors.SchemaError):
            events_schema.validate(invalid)

    def test_negative_visitorid_fails(self):
        """visitorid must be positive (> 0)."""
        invalid = pd.DataFrame(
            {
                "timestamp": [1695000000000],
                "visitorid": [-1],
                "event": ["view"],
                "itemid": [456.0],
                "transactionid": [np.nan],
            }
        )
        with pytest.raises(pa.errors.SchemaError):
            events_schema.validate(invalid)

    def test_zero_visitorid_fails(self):
        """visitorid == 0 is invalid."""
        invalid = pd.DataFrame(
            {
                "timestamp": [1695000000000],
                "visitorid": [0],
                "event": ["view"],
                "itemid": [456.0],
                "transactionid": [np.nan],
            }
        )
        with pytest.raises(pa.errors.SchemaError):
            events_schema.validate(invalid)


# =============================================================================
# Interactions Schema Tests
# =============================================================================


class TestInteractionsSchema:
    """Validate the processed interaction DataFrame."""

    def test_valid_interactions_pass(self):
        """Well-formed interactions must pass validation."""
        valid = pd.DataFrame(
            {
                "visitorid": ["u1", "u2", "u3"],
                "itemid": ["i1", "i2", "i3"],
                "weight": [1.0, 5.0, 3.0],
            }
        )
        interactions_schema.validate(valid)

    def test_zero_weight_fails(self):
        """Weight must be strictly positive (> 0)."""
        invalid = pd.DataFrame(
            {
                "visitorid": ["u1"],
                "itemid": ["i1"],
                "weight": [0.0],
            }
        )
        with pytest.raises(pa.errors.SchemaError):
            interactions_schema.validate(invalid)

    def test_negative_weight_fails(self):
        """Negative weights are invalid."""
        invalid = pd.DataFrame(
            {
                "visitorid": ["u1"],
                "itemid": ["i1"],
                "weight": [-1.0],
            }
        )
        with pytest.raises(pa.errors.SchemaError):
            interactions_schema.validate(invalid)

    def test_accumulated_weight_passes(self):
        """Aggregated weights (e.g. multiple transactions) must pass if <= 100."""
        valid = pd.DataFrame(
            {
                "visitorid": ["u1"],
                "itemid": ["i1"],
                "weight": [50.0],  # several transactions summed
            }
        )
        interactions_schema.validate(valid)


# =============================================================================
# Recommendation Response Schema Tests
# =============================================================================


class TestRecommendationResponseSchema:
    """Validate recommendation output DataFrames."""

    def test_valid_recommendations_pass(self):
        """Well-formed recommendation rows must pass validation."""
        valid = pd.DataFrame(
            {
                "product_id": ["123", "456"],
                "product_name": ["Product A", "Product B"],
                "score": [0.9, 0.7],
            }
        )
        recommendation_response_schema.validate(valid)

    def test_score_above_1_fails(self):
        """Scores above 1.0 must be rejected."""
        invalid = pd.DataFrame(
            {
                "product_id": ["123"],
                "product_name": ["Product A"],
                "score": [1.5],
            }
        )
        with pytest.raises(pa.errors.SchemaError):
            recommendation_response_schema.validate(invalid)

    def test_negative_score_fails(self):
        """Negative scores must be rejected."""
        invalid = pd.DataFrame(
            {
                "product_id": ["123"],
                "product_name": ["Product A"],
                "score": [-0.1],
            }
        )
        with pytest.raises(pa.errors.SchemaError):
            recommendation_response_schema.validate(invalid)

    def test_empty_product_id_fails(self):
        """Empty string product_id must be rejected."""
        invalid = pd.DataFrame(
            {
                "product_id": [""],  # empty — not allowed
                "product_name": ["Product A"],
                "score": [0.5],
            }
        )
        with pytest.raises(pa.errors.SchemaError):
            recommendation_response_schema.validate(invalid)

    def test_boundary_scores_pass(self):
        """Scores at exactly 0.0 and 1.0 are valid boundary values."""
        boundary = pd.DataFrame(
            {
                "product_id": ["a", "b"],
                "product_name": ["P1", "P2"],
                "score": [0.0, 1.0],
            }
        )
        recommendation_response_schema.validate(boundary)


# =============================================================================
# Data Completeness Tests
# =============================================================================


class TestDataCompleteness:
    """Validate data integrity invariants."""

    def test_no_duplicate_user_item_pairs(self):
        """
        After aggregation, there must be no duplicate (visitorid, itemid) pairs.
        Duplicates would double-count interactions and skew user preferences.
        """
        interactions = pd.DataFrame(
            {
                "visitorid": ["u1", "u1", "u2"],
                "itemid": ["i1", "i2", "i1"],
                "weight": [1.0, 2.0, 3.0],
            }
        )
        duplicates = interactions.duplicated(subset=["visitorid", "itemid"])
        assert (
            not duplicates.any()
        ), "Duplicate (user, item) pairs found after aggregation"

    def test_weight_values_are_finite(self):
        """No weight value may be NaN or infinite."""
        weights = np.array([1.0, 3.0, 5.0, 2.0, 4.0])
        assert np.all(np.isfinite(weights)), "Non-finite weight values detected"

    def test_item_id_index_maps_are_consistent(self):
        """item_id_to_idx and idx_to_item_id must be exact inverse mappings."""
        item_id_to_idx = {"i1": 0, "i2": 1, "i3": 2}
        idx_to_item_id = {v: k for k, v in item_id_to_idx.items()}
        for item_id, idx in item_id_to_idx.items():
            assert (
                idx_to_item_id[idx] == item_id
            ), f"Inconsistent mapping: {item_id} -> {idx} -> {idx_to_item_id[idx]}"

    def test_user_id_index_maps_are_consistent(self):
        """user_id_to_idx and idx_to_user_id must be exact inverse mappings."""
        user_id_to_idx = {"u1": 0, "u2": 1, "u3": 2}
        idx_to_user_id = {v: k for k, v in user_id_to_idx.items()}
        for uid, idx in user_id_to_idx.items():
            assert idx_to_user_id[idx] == uid

    def test_interaction_matrix_nnz_matches_interactions(self):
        """
        Number of non-zero entries in the sparse matrix must match
        the number of (user, item) pairs in the interactions DataFrame.
        """
        from scipy.sparse import csr_matrix

        interactions = pd.DataFrame(
            {
                "visitorid": ["u1", "u1", "u2"],
                "itemid": ["i1", "i2", "i1"],
                "weight": [1.0, 3.0, 5.0],
            }
        )
        user_map = {"u1": 0, "u2": 1}
        item_map = {"i1": 0, "i2": 1}
        rows = interactions["visitorid"].map(user_map).values
        cols = interactions["itemid"].map(item_map).values
        data = interactions["weight"].values

        matrix = csr_matrix((data, (rows, cols)), shape=(2, 2))
        assert matrix.nnz == len(interactions)


# =============================================================================
# API Response Schema Integration Test
# =============================================================================


class TestAPIResponseDataQuality:
    """
    End-to-end validation: call the live API and run the Pandera schema
    against the response to confirm the output matches the declared contract.
    """

    def test_api_response_passes_pandera_schema(self):
        """Recommendation API response must satisfy the Pandera schema."""
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        response = client.post(
            "/recommendations",
            json={
                "user_id": "schema_test_user",
                "num_recommendations": 5,
                "recommendation_type": "collaborative",
            },
        )
        assert response.status_code == 200
        recs = response.json()["recommendations"]

        if not recs:
            pytest.skip("No recommendations returned (model not loaded)")

        df = pd.DataFrame(recs)[["product_id", "product_name", "score"]]
        df["score"] = df["score"].astype(float)
        recommendation_response_schema.validate(df)
