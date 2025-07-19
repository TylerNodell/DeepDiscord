#!/usr/bin/env python3
"""
Test the consent management system
"""

import os
import sys
import tempfile
import json
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from discord_bot.discord_bot import ConsentManager

def test_consent_manager():
    """Test the ConsentManager class"""
    print("🧪 Testing Consent Manager")
    print("=" * 40)
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        consent_manager = ConsentManager(temp_dir)
        
        # Test initial state
        assert not consent_manager.has_consent(12345), "Should not have consent initially"
        print("✅ Initial state: No consent")
        
        # Test granting consent
        consent_manager.grant_consent(12345, 67890, expires_days=90)
        assert consent_manager.has_consent(12345), "Should have consent after granting"
        print("✅ Consent granted successfully")
        
        # Test consent info
        info = consent_manager.get_consent_info(12345)
        assert info is not None, "Should have consent info"
        assert info['status'] == 'granted', "Status should be granted"
        assert info['user_id'] == 12345, "User ID should match"
        assert info['granted_by_request_from'] == 67890, "Requester should match"
        print("✅ Consent info retrieved correctly")
        
        # Test consent persistence
        consent_manager.save_consents()
        new_manager = ConsentManager(temp_dir)
        assert new_manager.has_consent(12345), "Consent should persist after reload"
        print("✅ Consent persistence works")
        
        # Test consent revocation
        consent_manager.revoke_consent(12345)
        assert not consent_manager.has_consent(12345), "Should not have consent after revocation"
        info = consent_manager.get_consent_info(12345)
        assert info['status'] == 'revoked', "Status should be revoked"
        print("✅ Consent revocation works")
        
        # Test consent request creation
        request_id = consent_manager.create_consent_request(11111, 22222)
        assert len(request_id) == 8, "Request ID should be 8 characters"
        request = consent_manager.get_pending_request(request_id)
        assert request is not None, "Should be able to retrieve pending request"
        assert request['user_id'] == 11111, "User ID should match"
        assert request['requester_id'] == 22222, "Requester ID should match"
        print("✅ Consent request creation works")
        
        # Test completing request
        consent_manager.complete_request(request_id, True)
        assert consent_manager.has_consent(11111), "Should have consent after completing request"
        print("✅ Request completion works")
        
    print("\n🎉 All consent manager tests passed!")

def test_consent_expiration():
    """Test consent expiration functionality"""
    print("\n🧪 Testing Consent Expiration")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        consent_manager = ConsentManager(temp_dir)
        
        # Grant consent with short expiration
        consent_manager.grant_consent(12345, 67890, expires_days=1)
        assert consent_manager.has_consent(12345), "Should have consent"
        print("✅ Consent granted with expiration")
        
        # Manually set expiration to past
        consent_data = consent_manager.consents['12345']
        past_date = datetime.now() - timedelta(days=1)
        consent_data['expires_at'] = past_date.isoformat()
        
        assert not consent_manager.has_consent(12345), "Should not have consent after expiration"
        print("✅ Consent expiration works")
        
    print("\n🎉 Consent expiration tests passed!")

def test_consent_file_format():
    """Test the consent file format"""
    print("\n🧪 Testing Consent File Format")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        consent_manager = ConsentManager(temp_dir)
        
        # Grant some consents
        consent_manager.grant_consent(11111, 22222, expires_days=90)
        consent_manager.grant_consent(33333, 44444)
        consent_manager.revoke_consent(33333)
        
        # Check file format
        consent_file = os.path.join(temp_dir, "user_consents.json")
        assert os.path.exists(consent_file), "Consent file should exist"
        
        with open(consent_file, 'r') as f:
            data = json.load(f)
        
        assert '11111' in data, "Should have user 11111"
        assert '33333' in data, "Should have user 33333"
        
        user_11111 = data['11111']
        assert user_11111['status'] == 'granted', "User 11111 should be granted"
        assert 'granted_at' in user_11111, "Should have granted_at timestamp"
        assert 'expires_at' in user_11111, "Should have expires_at timestamp"
        
        user_33333 = data['33333']
        assert user_33333['status'] == 'revoked', "User 33333 should be revoked"
        assert 'revoked_at' in user_33333, "Should have revoked_at timestamp"
        
        print("✅ Consent file format is correct")
        
    print("\n🎉 Consent file format tests passed!")

def test_edge_cases():
    """Test edge cases and error conditions"""
    print("\n🧪 Testing Edge Cases")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        consent_manager = ConsentManager(temp_dir)
        
        # Test non-existent user
        assert not consent_manager.has_consent(99999), "Non-existent user should not have consent"
        info = consent_manager.get_consent_info(99999)
        assert info is None, "Non-existent user should have no info"
        print("✅ Non-existent user handled correctly")
        
        # Test non-existent request
        request = consent_manager.get_pending_request("nonexistent")
        assert request is None, "Non-existent request should return None"
        print("✅ Non-existent request handled correctly")
        
        # Test revoking non-existent consent
        consent_manager.revoke_consent(88888)  # Should not crash
        print("✅ Revoking non-existent consent handled gracefully")
        
        # Test granting consent without expiration
        consent_manager.grant_consent(77777, 88888)  # No expires_days
        assert consent_manager.has_consent(77777), "Should have consent without expiration"
        info = consent_manager.get_consent_info(77777)
        assert 'expires_at' not in info, "Should not have expires_at when no expiration set"
        print("✅ Consent without expiration works")
        
    print("\n🎉 Edge case tests passed!")

def main():
    """Run all consent system tests"""
    print("🚀 Testing Discord Consent System")
    print("=" * 50)
    
    tests = [
        test_consent_manager,
        test_consent_expiration,
        test_consent_file_format,
        test_edge_cases
    ]
    
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All consent system tests passed! The system is ready for deployment.")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")

if __name__ == "__main__":
    main()