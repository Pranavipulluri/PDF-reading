"""
Heading Context Tracker - Maintains hierarchical context and fixes outline gaps
Tracks heading transitions and ensures logical document structure
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

class HeadingContextTracker:
    """Tracks heading hierarchy and fixes structural inconsistencies"""
    
    def __init__(self, max_history: int = 50):
        self.logger = logging.getLogger(__name__)
        self.max_history = max_history
        
        # Hierarchy stack - maintains current heading context
        self.hierarchy_stack = deque(maxlen=max_history)
        
        # Statistics for analysis
        self.heading_stats = {
            'total_headings': 0,
            'level_counts': {'H1': 0, 'H2': 0, 'H3': 0},
            'transitions': [],
            'gaps_fixed': 0,
            'level_adjustments': 0
        }
        
        # Configuration
        self.config = {
            'allow_level_skipping': False,   # Strict hierarchy by default
            'auto_fix_gaps': True,           # Automatically fix gaps
            'max_consecutive_same_level': 10, # Max same-level headings
            'require_h1_start': False,       # Don't require H1 at start
        }
    
    def process_headings(self, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process headings to ensure proper hierarchy
        
        Args:
            headings: List of detected headings
            
        Returns:
            Processed headings with fixed hierarchy
        """
        if not headings:
            return headings
        
        self.logger.info(f"Processing {len(headings)} headings for hierarchy validation")
        
        # Reset state
        self.hierarchy_stack.clear()
        self.heading_stats = {
            'total_headings': len(headings),
            'level_counts': {'H1': 0, 'H2': 0, 'H3': 0},
            'transitions': [],
            'gaps_fixed': 0,
            'level_adjustments': 0
        }
        
        processed_headings = []
        
        for i, heading in enumerate(headings):
            # Process each heading in context
            processed_heading = self._process_single_heading(heading, i, headings)
            processed_headings.append(processed_heading)
            
            # Update statistics
            level = processed_heading.get('level', 'H3')
            self.heading_stats['level_counts'][level] = self.heading_stats['level_counts'].get(level, 0) + 1
        
        # Post-processing analysis
        self._analyze_document_structure(processed_headings)
        
        self.logger.info(f"Hierarchy processing complete: {self.heading_stats['gaps_fixed']} gaps fixed, "
                        f"{self.heading_stats['level_adjustments']} levels adjusted")
        
        return processed_headings
    
    def _process_single_heading(self, heading: Dict[str, Any], index: int, all_headings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a single heading in context"""
        
        processed_heading = heading.copy()
        original_level = heading.get('level', 'H3')
        current_level_num = self._level_to_number(original_level)
        
        # Analyze context
        context = self._analyze_current_context()
        
        # Check for hierarchy violations
        adjusted_level = self._check_and_fix_hierarchy(current_level_num, context, index)
        
        if adjusted_level != current_level_num:
            processed_heading['level'] = self._number_to_level(adjusted_level)
            processed_heading['hierarchy_adjusted'] = True
            processed_heading['original_level'] = original_level
            processed_heading['adjustment_reason'] = self._get_adjustment_reason(current_level_num, adjusted_level, context)
            self.heading_stats['level_adjustments'] += 1
        
        # Update hierarchy stack
        self._update_hierarchy_stack(processed_heading)
        
        # Record transition
        if self.hierarchy_stack and len(self.hierarchy_stack) > 1:
            prev_heading = list(self.hierarchy_stack)[-2]
            transition = {
                'from': prev_heading['level'],
                'to': processed_heading['level'],
                'page': processed_heading.get('page', 0)
            }
            self.heading_stats['transitions'].append(transition)
        
        return processed_heading
    
    def _analyze_current_context(self) -> Dict[str, Any]:
        """Analyze current hierarchy context"""
        
        context = {
            'stack_depth': len(self.hierarchy_stack),
            'current_levels': [],
            'last_level': None,
            'consecutive_same_level': 0,
            'has_h1': False,
            'has_h2': False
        }
        
        if not self.hierarchy_stack:
            return context
        
        # Extract current levels from stack
        stack_list = list(self.hierarchy_stack)
        context['current_levels'] = [h['level'] for h in stack_list]
        context['last_level'] = stack_list[-1]['level'] if stack_list else None
        
        # Count consecutive same-level headings
        if stack_list:
            last_level = stack_list[-1]['level']
            count = 1
            for i in range(len(stack_list) - 2, -1, -1):
                if stack_list[i]['level'] == last_level:
                    count += 1
                else:
                    break
            context['consecutive_same_level'] = count
        
        # Check for presence of different levels
        context['has_h1'] = 'H1' in context['current_levels']
        context['has_h2'] = 'H2' in context['current_levels']
        
        return context
    
    def _check_and_fix_hierarchy(self, current_level: int, context: Dict[str, Any], index: int) -> int:
        """Check and fix hierarchy violations"""
        
        if not context['last_level']:
            # First heading - allow any level but prefer logical start
            if not self.config['require_h1_start']:
                return current_level
            else:
                return 1  # Force H1 start
        
        last_level_num = self._level_to_number(context['last_level'])
        
        # Check for level skipping
        if current_level > last_level_num + 1 and not self.config['allow_level_skipping']:
            # Gap detected - fix it
            adjusted_level = last_level_num + 1
            self.heading_stats['gaps_fixed'] += 1
            self.logger.debug(f"Fixed hierarchy gap: {self._number_to_level(current_level)} → {self._number_to_level(adjusted_level)} at position {index}")
            return adjusted_level
        
        # Check for too many consecutive same-level headings
        if (current_level == last_level_num and 
            context['consecutive_same_level'] >= self.config['max_consecutive_same_level']):
            
            # Consider promoting to higher level or demoting
            if current_level > 1:
                return current_level - 1  # Promote to higher level
        
        return current_level
    
    def _update_hierarchy_stack(self, heading: Dict[str, Any]):
        """Update the hierarchy stack with new heading"""
        
        current_level = self._level_to_number(heading['level'])
        
        # Remove items from stack that are at same or lower hierarchy level
        while self.hierarchy_stack:
            top_level = self._level_to_number(self.hierarchy_stack[-1]['level'])
            if top_level >= current_level:
                self.hierarchy_stack.pop()
            else:
                break
        
        # Add current heading to stack
        self.hierarchy_stack.append({
            'level': heading['level'],
            'text': heading.get('text', ''),
            'page': heading.get('page', 0),
            'index': len(self.hierarchy_stack)
        })
    
    def _get_adjustment_reason(self, original_level: int, adjusted_level: int, context: Dict[str, Any]) -> str:
        """Get human-readable reason for level adjustment"""
        
        if adjusted_level < original_level:
            return f"promoted_to_maintain_hierarchy"
        elif adjusted_level > original_level:
            return f"demoted_to_maintain_hierarchy"
        else:
            return "no_change"
    
    def _analyze_document_structure(self, headings: List[Dict[str, Any]]):
        """Analyze overall document structure"""
        
        if not headings:
            return
        
        structure_analysis = {
            'total_headings': len(headings),
            'level_distribution': {},
            'max_depth': 0,
            'has_consistent_hierarchy': True,
            'common_issues': []
        }
        
        # Analyze level distribution
        for level in ['H1', 'H2', 'H3']:
            count = sum(1 for h in headings if h.get('level') == level)
            structure_analysis['level_distribution'][level] = count
        
        # Calculate max depth
        level_numbers = [self._level_to_number(h.get('level', 'H3')) for h in headings]
        structure_analysis['max_depth'] = max(level_numbers) if level_numbers else 0
        
        # Check for common issues
        h1_count = structure_analysis['level_distribution'].get('H1', 0)
        h2_count = structure_analysis['level_distribution'].get('H2', 0)
        h3_count = structure_analysis['level_distribution'].get('H3', 0)
        
        if h1_count == 0:
            structure_analysis['common_issues'].append('no_h1_headings')
        
        if h1_count > h2_count and h2_count > 0:
            structure_analysis['common_issues'].append('too_many_h1_relative_to_h2')
        
        if h3_count > h2_count * 3 and h2_count > 0:
            structure_analysis['common_issues'].append('too_many_h3_relative_to_h2')
        
        # Check for consistency
        transitions = self.heading_stats['transitions']
        irregular_transitions = [t for t in transitions 
                               if self._is_irregular_transition(t['from'], t['to'])]
        
        if len(irregular_transitions) > len(transitions) * 0.3:  # More than 30% irregular
            structure_analysis['has_consistent_hierarchy'] = False
            structure_analysis['common_issues'].append('irregular_hierarchy_transitions')
        
        self.heading_stats['structure_analysis'] = structure_analysis
        
        self.logger.info(f"Document structure: {h1_count} H1, {h2_count} H2, {h3_count} H3 headings")
        if structure_analysis['common_issues']:
            self.logger.info(f"Structure issues detected: {', '.join(structure_analysis['common_issues'])}")
    
    def _is_irregular_transition(self, from_level: str, to_level: str) -> bool:
        """Check if a level transition is irregular"""
        
        from_num = self._level_to_number(from_level)
        to_num = self._level_to_number(to_level)
        
        # Irregular transitions:
        # 1. Skipping levels downward (H1 -> H3)
        # 2. Going up more than one level (H3 -> H1)
        
        if to_num > from_num + 1:  # Skipping down
            return True
        
        if from_num > to_num + 1:  # Jumping up more than one level
            return True
        
        return False
    
    def _level_to_number(self, level: str) -> int:
        """Convert heading level string to number"""
        level_map = {'H1': 1, 'H2': 2, 'H3': 3, 'H4': 4, 'H5': 5, 'H6': 6}
        return level_map.get(level, 3)  # Default to H3
    
    def _number_to_level(self, number: int) -> str:
        """Convert number to heading level string"""
        number_map = {1: 'H1', 2: 'H2', 3: 'H3', 4: 'H4', 5: 'H5', 6: 'H6'}
        return number_map.get(number, 'H3')  # Default to H3
    
    def get_hierarchy_context(self) -> Dict[str, Any]:
        """Get current hierarchy context for debugging"""
        
        return {
            'current_stack': [
                {
                    'level': item['level'],
                    'text': item['text'][:50] + '...' if len(item['text']) > 50 else item['text'],
                    'page': item['page']
                }
                for item in self.hierarchy_stack
            ],
            'stack_depth': len(self.hierarchy_stack),
            'statistics': self.heading_stats.copy()
        }
    
    def generate_hierarchy_report(self) -> str:
        """Generate a human-readable hierarchy report"""
        
        report_lines = [
            "=== HEADING HIERARCHY REPORT ===",
            f"Total headings processed: {self.heading_stats['total_headings']}",
            ""
        ]
        
        # Level distribution
        report_lines.append("Level Distribution:")
        for level in ['H1', 'H2', 'H3']:
            count = self.heading_stats['level_counts'].get(level, 0)
            percentage = (count / max(self.heading_stats['total_headings'], 1)) * 100
            report_lines.append(f"  {level}: {count} ({percentage:.1f}%)")
        
        report_lines.append("")
        
        # Adjustments made
        if self.heading_stats['level_adjustments'] > 0:
            report_lines.append(f"Hierarchy adjustments made: {self.heading_stats['level_adjustments']}")
        
        if self.heading_stats['gaps_fixed'] > 0:
            report_lines.append(f"Hierarchy gaps fixed: {self.heading_stats['gaps_fixed']}")
        
        # Structure analysis
        if 'structure_analysis' in self.heading_stats:
            analysis = self.heading_stats['structure_analysis']
            
            if analysis['common_issues']:
                report_lines.append("")
                report_lines.append("Issues detected:")
                for issue in analysis['common_issues']:
                    issue_desc = issue.replace('_', ' ').title()
                    report_lines.append(f"  - {issue_desc}")
            
            if analysis['has_consistent_hierarchy']:
                report_lines.append("")
                report_lines.append("✓ Document has consistent hierarchy")
            else:
                report_lines.append("")
                report_lines.append("⚠ Document has inconsistent hierarchy")
        
        report_lines.append("")
        report_lines.append("=== END REPORT ===")
        
        return "\n".join(report_lines)
    
    def validate_final_hierarchy(self, headings: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate final heading hierarchy
        
        Args:
            headings: Final processed headings
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not headings:
            return True, issues
        
        prev_level = None
        h1_count = 0
        
        for i, heading in enumerate(headings):
            level = heading.get('level', 'H3')
            level_num = self._level_to_number(level)
            
            # Count H1s
            if level == 'H1':
                h1_count += 1
            
            # Check for level skipping
            if prev_level is not None:
                prev_level_num = self._level_to_number(prev_level)
                
                if level_num > prev_level_num + 1:
                    issues.append(f"Level skip detected at position {i}: {prev_level} → {level}")
            
            prev_level = level
        
        # Structural checks
        if h1_count == 0:
            issues.append("Document has no H1 headings")
        
        total_headings = len(headings)
        h2_count = sum(1 for h in headings if h.get('level') == 'H2')
        h3_count = sum(1 for h in headings if h.get('level') == 'H3')
        
        # Warn about unusual distributions
        if h1_count > total_headings * 0.5:
            issues.append("Too many H1 headings (>50% of total)")
        
        if h3_count > h2_count * 5 and h2_count > 0:
            issues.append("Disproportionate number of H3 headings relative to H2")
        
        is_valid = len(issues) == 0
        return is_valid, issues