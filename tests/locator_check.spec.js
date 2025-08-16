// @ts-check
const { test, expect } = require('@playwright/test');

// Change this URL if your HTML is hosted somewhere else
const LOCAL_PAGE = 'http://127.0.0.1:5500/html/sample_train.html';

test.describe('Sample Page Locator Test', () => {

  test.beforeEach(async ({ page }) => {
    await page.goto('http://127.0.0.1:5500/html/sample_train.html');
  });

  test('Click Submit button', async ({ page }) => {
    const submitBtn = page.locator("#this-is-submit-button");
    await expect(submitBtn).toBeVisible();
    await submitBtn.click();
  });

  test('Click Cancel button', async ({ page }) => {
    const cancelBtn = page.locator('button.btn.btn-secondary', { hasText: 'Cancel' });
    await expect(cancelBtn).toBeVisible();
    await cancelBtn.click();
  });

  test('Click Save Changes button', async ({ page }) => {
    const saveBtn = page.locator('button[data-action="save"]');
    await expect(saveBtn).toBeVisible();
    await saveBtn.click();
  });

  test('Click Confirm button inside section', async ({ page }) => {
    const confirmBtn = page.locator('section >> text=Confirm');
    await expect(confirmBtn).toBeVisible();
    await confirmBtn.click();
  });

  test('Click Close button in footer', async ({ page }) => {
    const closeBtn = page.locator("#this-is-submit-button");
    await expect(closeBtn).toBeVisible();
    await closeBtn.click();
  });

});
