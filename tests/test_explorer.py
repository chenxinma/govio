"""测试 RelationExplorer 大小写不敏感的关系探查"""

import pandas as pd

from govio.mcp.core.explorer import RelationExplorer


class TestRelationExplorerCaseInsensitive:
    """测试大小写不敏感的关系探查"""

    def test_foreign_key_different_case_column_names(self):
        """测试不同大小写的列名能正确匹配"""
        explorer = RelationExplorer()

        df1 = pd.DataFrame(
            {
                "CUST_ID": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
            }
        )

        df2 = pd.DataFrame(
            {
                "cust_id": [1, 2, 3],
                "order_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            }
        )

        relations = explorer.infer_foreign_keys(df1, "customers", df2, "orders")

        assert len(relations) == 1
        assert relations[0]["source_column"] == "CUST_ID"
        assert relations[0]["target_column"] == "cust_id"
        assert relations[0]["confidence"] == 1.0

    def test_foreign_key_uppercase_id_suffix(self):
        """测试大写 _ID 后缀能正确识别"""
        explorer = RelationExplorer()

        df1 = pd.DataFrame(
            {
                "ORDER_ID": ["A1", "A2", "A3"],
                "amount": [100, 200, 300],
            }
        )

        df2 = pd.DataFrame(
            {
                "order_id": ["A1", "A2", "A3", "A4"],
                "status": ["pending", "shipped", "delivered", "pending"],
            }
        )

        relations = explorer.infer_foreign_keys(df1, "orders", df2, "order_status")

        assert len(relations) == 1
        assert relations[0]["source_column"] == "ORDER_ID"
        assert relations[0]["target_column"] == "order_id"

    def test_foreign_key_camel_case_id(self):
        """测试驼峰写法 orderId 能正确识别"""
        explorer = RelationExplorer()

        df1 = pd.DataFrame(
            {
                "orderId": [101, 102, 103],
                "product": ["P1", "P2", "P3"],
            }
        )

        df2 = pd.DataFrame(
            {
                "order_id": [101, 102, 103],
                "customer": ["C1", "C2", "C3"],
            }
        )

        relations = explorer.infer_foreign_keys(df1, "items", df2, "orders")

        assert len(relations) == 1
        assert relations[0]["source_column"] == "orderId"
        assert relations[0]["target_column"] == "order_id"

    def test_partial_overlap_detected(self):
        """测试部分值重叠能被检测（重叠率 > 50%）"""
        explorer = RelationExplorer()

        df1 = pd.DataFrame(
            {
                "PRODUCT_ID": [1, 2, 3],
            }
        )

        df2 = pd.DataFrame(
            {
                "product_id": [2, 3, 4, 5],
            }
        )

        relations = explorer.infer_foreign_keys(df1, "products", df2, "inventory")

        assert len(relations) == 1
        assert relations[0]["confidence"] > 0.5

    def test_no_false_positive_different_values(self):
        """测试值不重叠时不产生误报"""
        explorer = RelationExplorer()

        df1 = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "name": ["A", "B", "C"],
            }
        )

        df2 = pd.DataFrame(
            {
                "user_id": [100, 200, 300],
                "email": ["a@x.com", "b@x.com", "c@x.com"],
            }
        )

        relations = explorer.infer_foreign_keys(df1, "users1", df2, "users2")

        assert len(relations) == 0

    def test_column_similarity_included_in_explore(self):
        """测试 explore() 输出包含 column_similarity 类型"""
        explorer = RelationExplorer()

        df1 = pd.DataFrame(
            {
                "customer_email": ["a@example.com", "b@example.com"],
                "name": ["Alice", "Bob"],
            }
        )

        df2 = pd.DataFrame(
            {
                "cust_email": ["a@example.com", "b@example.com"],
                "phone": ["123", "456"],
            }
        )

        relations = explorer.explore({"users": df1, "contacts": df2})

        similarity_relations = [
            r for r in relations if r.get("type") == "column_similarity"
        ]
        assert len(similarity_relations) > 0

        email_sim = next(
            (
                r
                for r in similarity_relations
                if "email" in r["column1"].lower() and "email" in r["column2"].lower()
            ),
            None,
        )
        assert email_sim is not None
        assert email_sim["similarity"] > 0.7
