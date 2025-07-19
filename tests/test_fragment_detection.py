#!/usr/bin/env python3
"""
Test cases for Discord message fragment detection and combining
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from discord_bot.discord_bot import MessageTracker

class TestFragmentDetection(unittest.TestCase):
    """Test cases for message fragment detection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tracker = MessageTracker()
        
    def create_mock_message(self, id, author_id, content, created_at):
        """Helper to create mock Discord message"""
        msg = MagicMock()
        msg.id = id
        msg.author.id = author_id
        msg.content = content
        msg.created_at = created_at
        msg.reference = None
        msg.channel.id = 123456
        return msg
    
    def test_is_potential_fragment_no_punctuation(self):
        """Test detection of fragments without ending punctuation"""
        base_time = datetime.utcnow()
        
        # Add first message without punctuation
        msg1 = self.create_mock_message(1, 1000, "Hey I wanted to ask", base_time)
        msg2 = self.create_mock_message(2, 1000, "about the new feature", base_time + timedelta(seconds=2))
        
        asyncio.run(self.tracker.add_message(msg1))
        
        # Check if second message is detected as fragment
        result = self.tracker.is_potential_fragment(msg2)
        self.assertTrue(result, "Should detect as fragment when previous message lacks punctuation")
    
    def test_is_potential_fragment_with_punctuation(self):
        """Test that messages with punctuation are not fragments"""
        base_time = datetime.utcnow()
        
        # Add first message with punctuation
        msg1 = self.create_mock_message(3, 2000, "Hello everyone.", base_time)
        msg2 = self.create_mock_message(4, 2000, "How are you?", base_time + timedelta(seconds=2))
        
        asyncio.run(self.tracker.add_message(msg1))
        
        # Check if second message is detected as fragment
        result = self.tracker.is_potential_fragment(msg2)
        self.assertFalse(result, "Should not detect as fragment when previous message has punctuation")
    
    def test_continuation_patterns(self):
        """Test detection of continuation pattern fragments"""
        base_time = datetime.utcnow()
        
        # Test various continuation patterns
        test_cases = [
            ("Previous message", "oh wait I forgot", True),
            ("Previous message", "and another thing", True),
            ("Previous message", "actually never mind", True),
            ("Previous message", "Hello there", False),
            ("Previous message", "I mean to say", True),
        ]
        
        for i, (prev_content, curr_content, expected) in enumerate(test_cases):
            msg1 = self.create_mock_message(100 + i*2, 3000, prev_content, base_time)
            msg2 = self.create_mock_message(101 + i*2, 3000, curr_content, base_time + timedelta(seconds=1))
            
            asyncio.run(self.tracker.add_message(msg1))
            result = self.tracker.is_potential_fragment(msg2)
            
            self.assertEqual(result, expected, 
                f"'{curr_content}' should {'be' if expected else 'not be'} detected as continuation")
    
    def test_time_window(self):
        """Test fragment detection time window"""
        base_time = datetime.utcnow()
        
        # Within timeout window
        msg1 = self.create_mock_message(200, 4000, "First part", base_time)
        msg2 = self.create_mock_message(201, 4000, "second part", base_time + timedelta(seconds=25))
        
        asyncio.run(self.tracker.add_message(msg1))
        result = self.tracker.is_potential_fragment(msg2)
        self.assertTrue(result, "Should detect fragment within 30-second window")
        
        # Outside timeout window
        msg3 = self.create_mock_message(202, 4000, "First part", base_time)
        msg4 = self.create_mock_message(203, 4000, "much later part", base_time + timedelta(seconds=35))
        
        # Clear buffer and test
        self.tracker.fragment_buffer.clear()
        asyncio.run(self.tracker.add_message(msg3))
        # Simulate timeout
        self.tracker.fragment_buffer[4000] = []
        result = self.tracker.is_potential_fragment(msg4)
        self.assertTrue(result, "Should treat as new message group after timeout")
    
    def test_different_users(self):
        """Test that fragments are tracked per user"""
        base_time = datetime.utcnow()
        
        # User 1 message
        msg1 = self.create_mock_message(300, 5000, "User 1 message", base_time)
        # User 2 message  
        msg2 = self.create_mock_message(301, 6000, "User 2 message", base_time + timedelta(seconds=1))
        
        asyncio.run(self.tracker.add_message(msg1))
        asyncio.run(self.tracker.add_message(msg2))
        
        # Check buffers are separate
        self.assertIn(5000, self.tracker.fragment_buffer)
        self.assertIn(6000, self.tracker.fragment_buffer)
        self.assertEqual(len(self.tracker.fragment_buffer[5000]), 1)
        self.assertEqual(len(self.tracker.fragment_buffer[6000]), 1)

if __name__ == '__main__':
    unittest.main()