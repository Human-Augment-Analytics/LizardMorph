// Cookie utilities for session management
export class CookieUtils {
  /**
   * Set a cookie with optional expiration
   */
  static setCookie(name: string, value: string, days?: number): void {
    let expires = "";
    if (days) {
      const date = new Date();
      date.setTime(date.getTime() + days * 24 * 60 * 60 * 1000);
      expires = `; expires=${date.toUTCString()}`;
    }
    document.cookie = `${name}=${value}${expires}; path=/; SameSite=Lax`;
  }

  /**
   * Get a cookie value by name
   */
  static getCookie(name: string): string | null {
    const nameEQ = `${name}=`;
    const ca = document.cookie.split(";");

    for (const cookie of ca) {
      let c = cookie;
      while (c.charAt(0) === " ") {
        c = c.substring(1, c.length);
      }
      if (c.indexOf(nameEQ) === 0) {
        return c.substring(nameEQ.length, c.length);
      }
    }
    return null;
  }

  /**
   * Delete a cookie by name
   */
  static deleteCookie(name: string): void {
    document.cookie = `${name}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;`;
  }

  /**
   * Check if cookies are available
   */
  static areCookiesEnabled(): boolean {
    try {
      const testCookie = "test_cookie";
      this.setCookie(testCookie, "test");
      const result = this.getCookie(testCookie) === "test";
      this.deleteCookie(testCookie);
      return result;
    } catch {
      return false;
    }
  }
}
