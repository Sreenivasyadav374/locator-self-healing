"""
Example Playwright test file to demonstrate the locator fixing capabilities.
This file contains various locator patterns that the tool can parse and fix.
"""

import pytest
from playwright.sync_api import Page, expect


def test_login_form(page: Page):
    """Test login functionality with various locator types."""
    # Navigate to login page
    page.goto("https://example.com/login")
    
    # Fill username field - using ID selector
    page.locator("#username").fill("testuser@example.com")
    
    # Fill password field - using name attribute
    page.locator("[name='password']").fill("testpassword123")
    
    # Click login button - using class selector (might fail)
    page.locator(".login-btn").click()
    
    # Wait for dashboard to load
    page.wait_for_selector("#dashboard")
    
    # Verify successful login
    expect(page.locator(".welcome-message")).to_be_visible()
    expect(page.locator(".welcome-message")).to_have_text("Welcome, testuser!")


def test_search_functionality(page: Page):
    """Test search feature with different locator approaches."""
    page.goto("https://example.com/search")
    
    # Search input using placeholder
    page.get_by_placeholder("Enter search terms...").fill("playwright testing")
    
    # Search button using text content (might fail if text changes)
    page.get_by_text("Search").click()
    
    # Wait for results
    page.wait_for_selector(".search-results")
    
    # Verify results are displayed
    expect(page.locator(".search-results")).to_be_visible()
    expect(page.locator(".result-item")).to_have_count(10)


def test_form_submission(page: Page):
    """Test form submission with various input types."""
    page.goto("https://example.com/contact")
    
    # Fill contact form
    page.locator("#contact-name").fill("John Doe")
    page.locator("#contact-email").fill("john@example.com")
    page.locator("#contact-subject").select_option("General Inquiry")
    page.locator("#contact-message").fill("This is a test message.")
    
    # Submit form - using CSS selector that might be fragile
    page.locator("form .submit-button").click()
    
    # Verify success message
    expect(page.locator(".success-alert")).to_be_visible()
    expect(page.locator(".success-alert")).to_have_text("Message sent successfully!")


def test_navigation_menu(page: Page):
    """Test navigation menu interactions."""
    page.goto("https://example.com")
    
    # Click on navigation items
    page.locator(".nav-menu .nav-item:has-text('Products')").click()
    expect(page).to_have_url("https://example.com/products")
    
    # Test dropdown menu
    page.locator(".nav-menu .dropdown-toggle").hover()
    page.locator(".dropdown-menu .menu-item:has-text('Services')").click()
    expect(page).to_have_url("https://example.com/services")


def test_dynamic_content(page: Page):
    """Test interactions with dynamically loaded content."""
    page.goto("https://example.com/dashboard")
    
    # Wait for dynamic content to load
    page.wait_for_selector("[data-testid='user-profile']")
    
    # Click on profile dropdown
    page.get_by_test_id("profile-dropdown").click()
    
    # Select settings option
    page.get_by_role("menuitem", name="Settings").click()
    
    # Verify settings page loaded
    expect(page).to_have_url("https://example.com/settings")
    expect(page.get_by_heading("User Settings")).to_be_visible()


def test_table_interactions(page: Page):
    """Test interactions with table elements."""
    page.goto("https://example.com/users")
    
    # Sort table by clicking header
    page.locator("th:has-text('Name')").click()
    
    # Filter table
    page.locator(".table-filter input").fill("john")
    page.locator(".table-filter .filter-btn").click()
    
    # Select first row
    page.locator("tbody tr:first-child .select-checkbox").check()
    
    # Perform bulk action
    page.locator(".bulk-actions select").select_option("delete")
    page.locator(".bulk-actions .apply-btn").click()
    
    # Confirm action
    page.locator(".confirmation-dialog .confirm-btn").click()


def test_modal_interactions(page: Page):
    """Test modal dialog interactions."""
    page.goto("https://example.com/products")
    
    # Open modal
    page.locator(".product-card:first-child .view-details").click()
    
    # Verify modal is open
    expect(page.locator(".modal")).to_be_visible()
    expect(page.locator(".modal-title")).to_have_text("Product Details")
    
    # Interact with modal content
    page.locator(".modal .quantity-input").fill("2")
    page.locator(".modal .add-to-cart-btn").click()
    
    # Close modal
    page.locator(".modal .close-btn").click()
    
    # Verify modal is closed
    expect(page.locator(".modal")).not_to_be_visible()


def test_file_upload(page: Page):
    """Test file upload functionality."""
    page.goto("https://example.com/upload")
    
    # Upload file
    page.locator("input[type='file']").set_input_files("test_file.txt")
    
    # Add description
    page.locator("#file-description").fill("Test file upload")
    
    # Submit upload
    page.locator(".upload-form .submit-btn").click()
    
    # Verify upload success
    expect(page.locator(".upload-success")).to_be_visible()
    expect(page.locator(".uploaded-file-name")).to_have_text("test_file.txt")


@pytest.mark.parametrize("user_type", ["admin", "user", "guest"])
def test_role_based_access(page: Page, user_type: str):
    """Test role-based access control."""
    page.goto(f"https://example.com/login?role={user_type}")
    
    # Login with role-specific credentials
    page.locator("#username").fill(f"{user_type}@example.com")
    page.locator("#password").fill("password123")
    page.locator(".login-submit").click()
    
    # Verify role-specific elements
    if user_type == "admin":
        expect(page.locator(".admin-panel")).to_be_visible()
        expect(page.locator(".user-management")).to_be_visible()
    elif user_type == "user":
        expect(page.locator(".user-dashboard")).to_be_visible()
        expect(page.locator(".admin-panel")).not_to_be_visible()
    else:  # guest
        expect(page.locator(".guest-welcome")).to_be_visible()
        expect(page.locator(".restricted-content")).not_to_be_visible()


def test_responsive_design(page: Page):
    """Test responsive design elements."""
    page.goto("https://example.com")
    
    # Test desktop view
    page.set_viewport_size({"width": 1200, "height": 800})
    expect(page.locator(".desktop-nav")).to_be_visible()
    expect(page.locator(".mobile-menu-toggle")).not_to_be_visible()
    
    # Test mobile view
    page.set_viewport_size({"width": 375, "height": 667})
    expect(page.locator(".desktop-nav")).not_to_be_visible()
    expect(page.locator(".mobile-menu-toggle")).to_be_visible()
    
    # Test mobile menu
    page.locator(".mobile-menu-toggle").click()
    expect(page.locator(".mobile-menu")).to_be_visible()