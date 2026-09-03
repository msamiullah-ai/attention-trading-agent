import React from 'react'

/**
 * Persistent, unmissable label rendered in the header of any card that is
 * currently showing mock/demo data instead of a live backend response. This
 * appears on every affected card individually — a single banner elsewhere
 * on the page isn't enough, since a person glancing at one widget has no
 * way to know its numbers are fake, and this is a surface that places real
 * orders.
 */
export const DemoBadge: React.FC = () => (
  <span className="demo-badge">DEMO DATA</span>
)
